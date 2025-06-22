#!/usr/bin/python3

import collections
import collections.abc
import copy
import dask.distributed as dd
import gc
import hdf5storage as mat
import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import pickle
import quantities as pq
import scipy
from statistics import median
import syncopy as spy
from tqdm import tqdm

from . import plotting

mne.set_log_level("CRITICAL")
NUM_WORKERS = os.cpu_count() * 3 // 4

class Signal(collections.abc.Sequence):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert len(data.shape) >= 2

        self._channels = channels
        self._data = data
        self._dt = dt
        self._timestamps = timestamps

    def cat_channels(self, other):
        assert self.__class__ == other.__class__
        assert self.data.units == other.data.units
        assert np.allclose(self.times, other.times)
        return self.__replace__(
            channels=pd.concat(self.channels, other.channels),
            data=np.concatenate((self.data, other.data), axis=0)
        )

    @property
    def channels(self):
        return self._channels

    @property
    def df(self):
        return 1. / self.T

    @property
    def dt(self):
        return self._dt

    def _data_slices(self, channels, times, trials):
        if channels is None:
            channels = slice(0, self.num_channels, 1)
        else:
            if isinstance(channels, int):
                channels = slice(channels, channels+1, None)
            if channels.step is None:
                channels = slice(channels.start, channels.stop, 1)
        if times is None:
            times = self.times.magnitude if hasattr(self.times, 'units')\
                    else self.times
            times = slice(times[0], times[-1], None)
        else:
            if isinstance(times, int):
                times = slice(times, times+1, None)
            if times.step is None:
                times = slice(times.start, times.stop, 1)
        if trials is None:
            trials = slice(None, None, None)
        else:
            if isinstance(trials, int):
                trials = slice(trials, trials+1, None)
            if trials.step is None:
                trials = slice(trials.start, trials.stop, 1)
        return channels, times, trials

    @property
    def f0(self):
        return 1. / self.dt

    def fmap(self, f):
        return self.__replace__(data=f(self.data))

    @property
    def fNQ(self):
        return self.f0 / 2.

    def get_data(self, channels=None, times=None, trials=None):
        assert isinstance(channels, slice)
        assert isinstance(times, slice)
        assert isinstance(trials, slice)
        raise NotImplementedError

    def __len__(self):
        return len(self._timestamps)

    @property
    def num_channels(self):
        return len(self._channels)

    @property
    def num_trials(self):
        raise NotImplementedError

    def __replace__(self, /, **changes):
        parameters = {field: changes.get(field, getattr(self, field)) for field
                      in ["channels", "data", "dt", "times"]}
        return self.__class__(*parameters.values())

    def sample_at(self, t):
        if hasattr(self._timestamps, "units") and not hasattr(t, "units"):
            times = self._timestamps.magnitude
        else:
            times = self._timestamps
        return np.nanargmin(np.abs(times - t))

    def sort_channels(self, key):
        indices = self.channels.sort_values(key, ascending=False).index
        return [self.channels.index.get_loc(i) for i in indices]

    @property
    def T(self):
        return self.dt * len(self)

    @property
    def times(self):
        return self._timestamps

    def time_to_samples(self, t):
        return math.ceil(t * self.f0)

class EpochedSignal(Signal):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert len(data.shape) >= 3
        assert len(channels) == data.shape[0]
        assert len(timestamps) == data.shape[1]
        assert hasattr(data, "units")
        assert timestamps.units == dt.units

        super().__init__(channels, data, dt, timestamps)

    def __add__(self, sig):
        assert self.__class__ == sig.__class__
        assert (self.channels == sig.channels).all().all()
        assert np.isclose(self.dt.magnitude, sig.dt.magnitude)

        num_samples = min(self.data.shape[1], sig.data.shape[1])
        if not np.allclose(self.times[:num_samples].magnitude,
                           sig.times[:num_samples].magnitude,
                           atol=self.dt.magnitude):
            dt = self.dt.magnitude if hasattr(self.dt, "units") else self.dt
            timestamps = pq.Quantity(np.arange(0, num_samples * dt, dt),
                                     self.dt.units)
        else:
            timestamps = self.times
        timestamps = timestamps[:num_samples]

        data = self.data[:, :num_samples] + sig.data[:, :num_samples]
        return self.__replace__(data=data, times=timestamps)

    def baseline_correct(self, start, stop):
        start, stop = self.sample_at(start), self.sample_at(stop) - 1
        def f(data):
            return data - data[:, start:stop].mean(axis=1)[:, np.newaxis, :]
        return self.fmap(f)

    def band_phases(self, fmin, fmax):
        fmin, fmax = fmin.rescale("Hz"), fmax.rescale("Hz")
        assert self.df.rescale("Hz") <= fmin
        cluster = dd.LocalCluster(n_workers=NUM_WORKERS)
        client = dd.Client(cluster)

        channels = [str(ch) for ch in list(self.channels.index.values)]
        middle = len(self.channels) // 2
        channels = channels[middle:middle+1]
        phases = []
        for c in tqdm(range(0, self.num_trials, NUM_WORKERS)):
            trials = slice(c, c + NUM_WORKERS)
            trial_xs = mne.EpochsArray(
                np.moveaxis(self.data.magnitude[middle:middle+1, :, trials],
                            -1, 0),
                mne.create_info(channels, int(self.f0.item())), proj=False,
            )

            data = spy.mne_epochs_to_tldata(trial_xs)
            cfg = spy.get_defaults(spy.preprocessing)
            cfg.filter_class = 'firws'
            cfg.filter_type = "bp" # Band-pass filter
            cfg.freq = [fmin.item(), fmax.item()]
            cfg.hilbert = "angle"
            cfg.parallel = None
            cfg.polyremoval = 0
            cfg.window = 'hann'
            phase = spy.preprocessing(cfg, data).show()
            if isinstance(phase, list):
                phase = np.stack(phase, axis=-1)
            else:
                phase = phase[:, np.newaxis]
            phases.append(phase)

            del data
            del trial_xs
            spy.cleanup(interactive=False)

        client.close()
        del client
        cluster.close()
        del cluster

        phases = np.concatenate(phases, axis=-1)[np.newaxis, :, :]
        from .signals.phase import EpochedPhase

        return EpochedPhase(self.channels[middle:(middle + 1)],
                            pq.Quantity(phases, pq.rad), self.dt,
                            self.times)

    def cat_trials(self, other):
        assert self.__class__ == other.__class__
        assert self.data.units == other.data.units
        assert np.allclose(self.times.magnitude, other.times.magnitude,
                           atol=self.dt.magnitude)
        return self.__replace__(
            data=np.concatenate((self.data, other.data), axis=-1) *\
                 self.data.units
        )

    @property
    def data(self):
        return self._data

    def detrend(self):
        erp = pq.Quantity(self.data.magnitude.mean(-1, keepdims=True),
                          self.data.units)
        return self.__replace__(data=self.data - erp)

    def downsample(self, n):
        channels = self.channels.loc[0::n]
        data = self.data[0::n, :, :]
        return self.__replace__(channels=channels, data=data)

    def epoch(self, intervals, time_shift=0.):
        assert intervals.shape == (self.num_trials, 2)
        if not hasattr(time_shift, "units"):
            time_shift = pq.Quantity(time_shift, self.dt.units)

        data = []
        for trial, (start, end) in enumerate(intervals):
            first, last = self.sample_at(start), self.sample_at(end)
            data.append(self._data[:, first:last, trial])
        time_length = min([trial.shape[1] for trial in data])
        units = data[0].units
        data = np.stack([trial[:, :time_length] for trial in data], axis=-1)
        timestamps = np.arange(data.shape[1]) * self.dt + time_shift
        return self.__replace__(data=data * units, times=timestamps)

    def epoch_oscillations(self, period, phases, phase=np.pi):
        period_samples = int(np.round(period / self.dt).item())
        coordinates = np.stack(np.isclose(np.abs(phases.data.magnitude), phase,
                                          atol=1e-3).nonzero(),
                               axis=-1)
        coordinates = pd.DataFrame(coordinates, columns=["channel", "sample",
                                                         "trial"])
        coordinates = coordinates.sort_values(by="trial")
        oscillations = []
        detrended = self.data - self.data.mean(axis=-1)[:, :, np.newaxis]
        for i in coordinates.index:
            channel, sample, trial = coordinates.values[i]
            first = self.sample_at(self.times[sample] - period / 2)
            last = self.sample_at(self.times[sample] + period / 2)
            if np.isclose(self.times[last] - self.times[first], period,
                          atol=self.dt).item():
                oscillations.append(detrended[:, first:last, trial])
        length = min([oscillation.shape[1] for oscillation in oscillations])
        oscillations = np.stack([oscillation[:, :length] for oscillation in
                                 oscillations], axis=-1) * oscillations[0].units
        timestamps = (np.arange(length) - period / (2  * self.dt)) * self.dt
        return self.__replace__(data=oscillations, times=timestamps)

    def evoked(self):
        data = pq.Quantity(self.data.magnitude.mean(-1, keepdims=True),
                           self.data.units)
        return EvokedSignal(self.channels, data, self.dt, self.times)

    def get_data(self, channels, times, trials):
        channels, times, trials = self._data_slices(channels, times, trials)

        times = slice(self.sample_at(times.start),
                      self.sample_at(times.stop) + 1, times.step)
        return self._data[channels, times, trials]

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if key.step is None:
            key = slice(key.start, key.stop, 1)

        key = slice(self.sample_at(key.start), self.sample_at(key.stop),
                    key.step)
        return self.__replace__(data=self.data[:, key], times=self.times[key])

    def mask_epochs(self, onsets, offsets):
        assert len(onsets) == len(offsets)

        self._data = np.nan_to_num(self._data, copy=False)
        for trial in range(len(onsets)):
            first = self.sample_at(onsets[trial])
            last = self.sample_at(offsets[trial])
            self._data[:, :first, trial] *= 0
            self._data[:, last:, trial] *= 0

    def median_filter(self, cs=3):
        def units_medfilt(data):
            result = scipy.ndimage.median_filter(data, size=(cs, 1, 1))
            return result * data.units
        return self.fmap(units_medfilt)

    def __mul__(self, sig):
        assert self.__class__ == sig.__class__
        assert (self.channels == sig.channels).all().all()
        assert np.isclose(self.dt.magnitude, sig.dt.magnitude)

        num_samples = min(self.data.shape[1], sig.data.shape[1])
        if not np.allclose(self.times[:num_samples].magnitude,
                           sig.times[:num_samples].magnitude,
                           atol=self.dt.magnitude):
            dt = self.dt.magnitude if hasattr(self.dt, "units") else self.dt
            timestamps = pq.Quantity(np.arange(0, num_samples * dt, dt),
                                     self.dt.units)
        else:
            timestamps = self.times
        timestamps = timestamps[:num_samples]

        data = self.data[:, :num_samples] * sig.data[:, :num_samples]
        return self.__replace__(data=data, times=timestamps)

    @property
    def num_channels(self):
        return self.data.shape[0]

    @property
    def num_trials(self):
        return self.data.shape[-1]

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        self.channels.to_csv(path + '/channels.csv')

        mat.savemat(path + '/epoched_signal.mat', {
            "data": self.data.magnitude, "timestamps": self.times.magnitude
        })
        other = copy.copy(self)
        other._channels = other._data = other._timestamps = None
        other._units = {"data": self.data.units, "timestamps": self.times.units}
        with open(path + "/epoched_signal.pickle", mode="wb") as f:
            pickle.dump(other, f)

    def select_channels(self, mask):
        channels = self.channels.copy()
        if "channel" not in channels.columns:
            channels.insert(len(channels.columns), "channel",
                            list(range(len(self.channels))))
        return self.__replace__(channels=channels.loc[mask],
                                data=self.data[mask, :])

    def select_trials(self, trials):
        trials = trials + [False] * (self.num_trials - len(trials))
        return self.__replace__(data=self.data[..., trials])

    def shift_timestamps(self, offset):
        return self.__replace__(times=self.times + offset)

    def significance_threshold(self, axis=-1, sems=2):
        std = self.data.magnitude.astype("float64").std(axis=axis,
                                                        keepdims=True)
        return sems * std / self.data.shape[axis]

    def __sub__(self, sig):
        assert self.__class__ == sig.__class__
        assert (self.channels == sig.channels).all().all()
        assert np.isclose(self.dt.magnitude, sig.dt.magnitude)

        num_samples = min(self.data.shape[1], sig.data.shape[1])
        if not np.allclose(self.times[:num_samples].magnitude,
                           sig.times[:num_samples].magnitude,
                           atol=self.dt.magnitude):
            dt = self.dt.magnitude if hasattr(self.dt, "units") else self.dt
            timestamps = pq.Quantity(np.arange(0, num_samples * dt, dt),
                                     self.dt.units)
        else:
            timestamps = self.times
        timestamps = timestamps[:num_samples]

        data = self.data[:, :num_samples] - sig.data[:, :num_samples]
        return self.__replace__(data=data, times=timestamps)

    def __truediv__(self, sig):
        assert self.__class__ == sig.__class__
        assert (self.channels == sig.channels).all().all()
        assert np.isclose(self.dt.magnitude, sig.dt.magnitude)

        num_samples = min(self.data.shape[1], sig.data.shape[1])
        if not np.allclose(self.times[:num_samples].magnitude,
                           sig.times[:num_samples].magnitude,
                           atol=self.dt.magnitude):
            dt = self.dt.magnitude if hasattr(self.dt, "units") else self.dt
            timestamps = pq.Quantity(np.arange(0, num_samples * dt, dt),
                                     self.dt.units)
        else:
            timestamps = self.times
        timestamps = timestamps[:num_samples]

        data = self.data[:, :num_samples] / sig.data[:, :num_samples]
        return self.__replace__(data=data, times=timestamps)

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + "/epoched_signal.pickle", mode="rb") as f:
            self = pickle.load(f)

        arrays = mat.loadmat(path + '/epoched_signal.mat')
        self._timestamps = pq.Quantity(arrays['timestamps'],
                                       self._units["timestamps"])
        self._data = pq.Quantity(arrays['data'], self._units["data"])
        self._channels = pd.read_csv(path + '/channels.csv', index_col=0)
        del arrays
        del self._units
        gc.collect()
        return self

def trials_ttest(sa: EpochedSignal, sb: EpochedSignal, pvalue=0.05):
    assert isinstance(sa, EpochedSignal)
    assert sa.__class__ == sb.__class__
    assert (sa.channels == sb.channels).all().all()
    assert sa.dt == sb.dt

    num_samples = min(sa.data.shape[1], sb.data.shape[1])
    np.allclose(sa.times[:num_samples], sb.times[:num_samples], sa.dt, sb.dt)
    timestamps = (sa.times[:num_samples] + sb.times[:num_samples]) / 2
    ttest = scipy.stats.ttest_ind(sa.data[:, :num_samples],
                                  sb.data[:, :num_samples], axis=-1,
                                  equal_var=False, keepdims=True)
    data = sa.data[:, :num_samples].mean(axis=-1, keepdims=True) -\
           sb.data[:, :num_samples].mean(axis=-1, keepdims=True)
    data *= ttest.pvalue < pvalue
    return sa.__class__(sa.channels, data, sa.dt, timestamps).evoked()

class EvokedSignal(EpochedSignal):
    def __init__(self, channels, data, dt, timestamps):
        assert data.shape[-1] == 1
        super().__init__(channels, data, dt, timestamps)

    def annotate_channels(self, ax, key, ycolumn=None):
        channels = [chan.decode() if isinstance(chan, bytes) else chan
                    for chan in self.channels[key].values]
        area = os.path.commonprefix(channels)
        laminar_channels = collections.defaultdict(lambda: [])
        for c, chan in enumerate(self.channels[key].values):
            layer = chan.removeprefix(area)
            layer = chan if "" in [layer, area] else 'L' + layer
            if ycolumn is not None:
                channel_y = self.channels[ycolumn].values[c]
            else:
                channel_y = c
            laminar_channels[layer].append(channel_y)

        xmin, xmax = ax.get_xbound()
        crossings = [max(laminar_channels[layer][0] - 1, 0) for layer
                     in laminar_channels.keys()]
        ax.hlines(crossings, xmin, xmax, linestyles=":")
        ax.set_yticks([], [])

        minortick_locs, laminar_labels = [], []
        for layer in laminar_channels:
            minortick_locs.append(median(laminar_channels[layer]))
            laminar_labels.append(layer)
        ax.set_yticks(minortick_locs, laminar_labels, minor=True)

    def line_plot(self, ax=None, fig=None, logspace=False, callback=None,
                  title=None, vmin=None, vmax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        data = self.data.T.squeeze()
        if logspace:
            data = np.where(data < 0., -np.log(-data), np.log(data))
        ax.plot(self.times, data)
        if title is not None:
            ax.set_title(title)
        if vmin is not None or vmax is not None:
            ax.set_ylim(vmin, vmax)

        if hasattr(self.times, 'units'):
            unit = list(self.times.units.dimensionality.keys())[0].name
            ax.set_xlabel((unit + 's').capitalize())
        if "location" in self.channels.columns:
            locations = [chan.decode() if isinstance(chan, bytes) else chan
                         for chan in self.channels["location"].values]
            ax.legend(locations)
        if callback is not None:
            callback(self, ax)

    def heatmap(self, alpha=None, ax=None, fig=None, filename=None, title=None,
                vmin=None, vmax=None, origin="lower", channel_ticks="location",
                baseline=None, cmap=None, callback=None, cbar=False):
        if ax is None:
            ax = plt.gca()
        figure = plt.gcf() if fig is None else fig

        data = self.data.squeeze()
        if alpha is not None:
            alpha = alpha.squeeze()
        img = plotting.heatmap(figure, ax, data, alpha=alpha, title=title,
                               vmin=vmin, vmax=vmax, cmap=cmap,
                               cbar=cbar or (vmin is None and vmax is None))

        times = self.times.rescale('ms')
        num_xticks = len(ax.get_xticks())
        xtick_locs = np.linspace(0, data.shape[1], num_xticks)
        xticks = np.linspace(times[0], times[-1], num_xticks)
        xticks = ["%0.2f" % t for t in xticks]
        ax.set_xticks(xtick_locs, xticks)
        ax.set_xlabel("Time (%s)" % times.units.dimensionality.string)
        if channel_ticks is not None and channel_ticks in self.channels.columns:
            self.annotate_channels(ax, channel_ticks)

        if baseline is not None:
            bxmin = self.sample_at(baseline[0])
            bxmax = self.sample_at(baseline[1])
            ax.axvspan(bxmin, bxmax, alpha=0.1, color='k')

        if callback is not None:
            callback(self, ax)

        if filename is not None:
            figure.savefig(filename)
            if fig is None:
                plt.close(figure)
        return img

    def plot(self, *args, **kwargs):
        if "events" in kwargs:
            events = kwargs.pop("events")
            def callback(self, ax):
                for (event, (time, color)) in events.items():
                    ymin, ymax = ax.get_ybound()
                    xtime = self.sample_at(time)
                    ax.vlines(xtime, ymin, ymax, colors=color,
                              linestyles='dashed', label=event)
                    ax.annotate(event, (xtime + 0.005, ymax))
            kwargs["callback"] = callback
        return self.line_plot(*args, **kwargs)

class RawSignal(Signal):
    epoched_signal = EpochedSignal

    def __init__(self, channels: pd.DataFrame, data, dt, timestamps,
                 channels_dim=0, time_dim=1):
        assert len(data.shape) == 2
        assert len(channels) == data.shape[channels_dim]
        assert len(timestamps) == data.shape[time_dim]
        assert hasattr(data, "units")
        assert timestamps.units == dt.units

        self._channels_dim = channels_dim
        self._time_dim = time_dim
        super().__init__(channels, data, dt, timestamps)

    def epoch(self, intervals, time_shift=0.):
        assert intervals.shape[1] == 2 and intervals.shape[0] >= 1

        trials_data = []
        for trial, (start, end) in enumerate(intervals):
            trials_data.append(self[start:end])
        trials_samples = min(data.shape[1] for data in trials_data)
        trials_data = [data[:, :trials_samples] for data in trials_data]
        units = trials_data[0].units
        trials_data = np.concatenate(trials_data, axis=-1) * units
        timestamps = np.arange(trials_data.shape[1]) * self.dt + time_shift
        return self.epoched_signal(self.channels, trials_data, self.dt,
                                   timestamps)

    def get_data(self, channels, times, trials):
        channels, times, trials = self._data_slices(channels, times, trials)
        assert trials == slice(None, None, None)

        times = slice(self.sample_at(times.start), self.sample_at(times.stop),
                      times.step)

        keys = [None, None]
        keys[self._channels_dim] = channels
        keys[self._time_dim] = times
        data = self._data[keys[0], keys[1]]
        data = data.swapaxes(self._channels_dim, 0)
        return data[:, :, np.newaxis]

    def __getitem__(self, key):
        return self.get_data(None, key, None)

    @property
    def num_trials(self):
        return 1
