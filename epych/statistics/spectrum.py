#!/usr/bin/python3

import collections
import dask.array
import dask.distributed as dd
import fooof
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import quantities as pq
import scipy.fft as fft
from statistics import median
import syncopy as spy
from tqdm import tqdm

from .. import plotting, signal, signals, statistic

mne.set_log_level("CRITICAL")

THETA_BAND = (2. * pq.Hz, 8. * pq.Hz)
ALPHA_BAND = (8. * pq.Hz, 15 * pq.Hz)
BETA_BAND = (15. * pq.Hz, 30 * pq.Hz)
ALPHA_BETA_BAND = (ALPHA_BAND[0], BETA_BAND[1])
LOW_GAMMA_BAND = (30. * pq.Hz, 50. * pq.Hz)
HIGH_GAMMA_BAND = (50 * pq.Hz, 90. * pq.Hz)
GAMMA_BAND = (LOW_GAMMA_BAND[0], HIGH_GAMMA_BAND[1])
HIGH_FREQUENCY_BAND = (90. * pq.Hz, 150. * pq.Hz)

decibel = pq.UnitQuantity(
    'decibel',
    0.1 * pq.dimensionless,
    symbol='dB',
    aliases=['dBs']
)

NUM_WORKERS = os.cpu_count() * 3 // 4
cluster = dd.LocalCluster(n_workers=NUM_WORKERS)
client = dd.Client(cluster)

class PowerSpectrum(statistic.ChannelwiseStatistic[signal.EpochedSignal]):
    def __init__(self, df, channels, f0, fmax=150, freqs=None, taper=None,
                 data=None):
        if not hasattr(fmax, "units"):
            fmax = np.array(fmax) * pq.Hz
        self._df = df.rescale("Hz")
        self._f0 = f0.rescale("Hz")
        if freqs is None:
            self._freqs = np.arange(0, fmax.item(), df.item())
            self._freqs = (self._freqs + df.item()) * df.units
        else:
            self._freqs = freqs
        self._taper = taper
        super().__init__(channels, (len(self._freqs),), data=data)

    def annotate_channels(self, ax, key, ycolumn=None):
        channels = [chan.decode() if isinstance(chan, bytes) else chan
                    for chan in self.channels[key].values]
        area = os.path.commonprefix(channels)
        laminar_channels = collections.defaultdict(lambda: [])
        for c, chan in enumerate(self.channels[key].values):
            layer = 'L' + chan.removeprefix(area)
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

    def apply(self, element: signal.EpochedSignal):
        assert (element.channels.location == self.channels.location).all().all()
        assert np.isclose(element.df.magnitude, self.df.magnitude)
        assert element.f0.magnitude >= self.f0.magnitude

        channels = [str(ch) for ch in list(self.channels.index.values)]
        xs = element.data.magnitude - element.data.magnitude.mean(axis=-1,
                                                                  keepdims=True)
        xs = mne.EpochsArray(np.moveaxis(xs, -1, 0),
                             mne.create_info(channels, int(self.f0.item())),
                             proj=False)

        data = spy.mne_epochs_to_tldata(xs)
        cfg = spy.get_defaults(spy.freqanalysis)

        cfg.foi = self.freqs.magnitude.squeeze()
        cfg.ft_compat = True
        cfg.keeptrials = 'yes'
        cfg.method = 'mtmfft'
        cfg.output = 'pow'
        cfg.parallel = True
        cfg.polyremoval = 0
        cfg.taper = self._taper
        psd = np.stack(spy.freqanalysis(cfg, data).show(), axis=-1)
        psd = np.moveaxis(psd, 0, 1)

        del xs
        del data

        if self.data is None:
            return psd
        return np.concatenate((self.data, psd), axis=-1)

    def band_power(self, fbottom, ftop):
        ibot = np.nanargmin((self.freqs - fbottom) ** 2)
        itop = np.nanargmin((self.freqs - ftop) ** 2)
        return self.data[:, ibot:itop+1].mean(axis=1)

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    def decibels(self):
        return self.fmap(lambda vals: 10 * np.log10(vals))

    def density(self):
        normalizer = self.data.sum(axis=1)[:, np.newaxis, ...]
        return self.fmap(lambda vals: vals / normalizer)

    @property
    def df(self):
        return self._df

    @property
    def dt(self):
        return (1. / self.f0).rescale('s')

    def evoked(self):
        return self.fmap(lambda vals: vals.mean(axis=-1))

    def fmap(self, f):
        return self.__class__(self.df, self.channels, self.f0,
                              data=f(self.data), fmax=self.fmax,
                              freqs=self.freqs)

    @property
    def f0(self):
        return self._f0

    @property
    def fmax(self):
        return self._freqs[-1]

    @property
    def freqs(self):
        return self._freqs

    def heatmap(self, ax=None, channel_ticks="location", fig=None,
                filename=None, subtitle=None, title="Power Spectral Density",
                **kwargs):
        if fig is None:
            figure = plt.figure(dpi=100)
        else:
            figure = fig
        if ax is None:
            axes = figure.add_subplot()
        else:
            axes = ax
        if subtitle:
            title += " (%s)" % subtitle

        plotting.heatmap(figure, axes, self.data.magnitude, title=title,
                         **kwargs)
        axes.set_xlim(0, len(self.freqs))
        xticks = [int(xtick) for xtick in axes.get_xticks()]
        xtick_freqs = ["%.1f" % self.freqs[xtick] for xtick in xticks
                       if xtick < len(self.freqs)] + ["%.1f" % self.freqs[-1]]
        axes.set_xticks(xticks, xtick_freqs)
        axes.set_xlabel("Frequency (Hz)")

        if channel_ticks is not None and channel_ticks in self.channels.columns:
            self.annotate_channels(axes, channel_ticks)

        if filename:
            figure.savefig(filename, dpi=100)
        if fig is None:
            plt.show()
            plt.close(figure)

    def plot_channels(self, stat, ax=None, xlims=None):
        if ax is None:
            ax = plt.gca()

        channels = np.arange(0, self.data.shape[0])
        ax.plot(stat, channels)
        if xlims is not None:
            ax.set_xlim(*xlims)
        ax.invert_yaxis()

    def relative(self):
        max_pow = self.data.max(axis=0)[np.newaxis, ...]
        return self.fmap(lambda vals: vals / max_pow)

    def result(self):
        return self.data.mean(axis=-1)

    def oscillatory(self, channel_mean=True, mode="knee"):
        if channel_mean:
            fm = fooof.FOOOF(verbose=False, aperiodic_mode=mode)
            fm.fit(self.freqs, self.data.magnitude.mean(0).mean(-1),
                   (self.freqs[0], self.freqs[-1]))
            spec = self.select_freqs(fm.freqs[0], fm.freqs[-1])
            aperiodic = fm.get_model(component='aperiodic', space='linear')
            aperiodic = aperiodic[np.newaxis, :, np.newaxis]
            return (spec.fmap(lambda data: data / aperiodic * data.units),
                    fm.freqs, aperiodic)
        else:
            fg = fooof.FOOOFGroup(verbose=False, aperiodic_mode=mode)
            powers = self.data.magnitude.mean(axis=-1, keepdims=False)
            fg.fit(self.freqs, powers, freq_range=(self.freqs[0],
                   self.freqs[-1]), n_jobs=-1)

            aperiodic = []
            for chan in range(len(fg.get_results())):
                fm = fg.get_fooof(chan)
                aperiodic.append(fm.get_model('aperiodic', 'linear'))
            aperiodic = np.stack(aperiodic, axis=0)[:, :, np.newaxis]
            spec = self.select_freqs(fg.freqs[0], fg.freqs[-1])
            return (spec.fmap(lambda data: data / aperiodic * data.units),
                    fg.freqs, aperiodic)

    def select_freqs(self, low, high):
        low_idx = np.argmin(np.abs(self.freqs - low))
        high_idx = np.argmin(np.abs(self.freqs - high)) + 1
        return self.__class__(self.df, self.channels, self.f0,
                              data=self.data[:, low_idx:high_idx, :],
                              fmax=self.fmax,
                              freqs=self.freqs[low_idx:high_idx])

class Spectrogram(statistic.ChannelwiseStatistic[signal.EpochedSignal]):
    def __init__(self, df, channels, f0, fmax=120, taper=None, data=None,
                 path=None, time_window=0.250):
        if not hasattr(fmax, "units"):
            fmax = np.array(fmax) * pq.Hz
        self._df = (max(df.magnitude, 1 / time_window) * df.units).rescale("Hz")
        self._f0 = f0.rescale("Hz")
        self._freqs = np.arange(0, fmax.item() + 1, self.df.magnitude) * self.df.units
        self._k = 0
        self._taper = taper
        self._time_window = time_window
        self._path = path
        super().__init__(channels.copy(), (int((fmax / df).item()),), data=data)

    def apply(self, element: signal.EpochedSignal):
        assert (element.channels == self.channels).all().all()
        assert element.df.rescale("Hz") <= self.df
        assert element.f0 >= self.f0

        element_data = []
        channels = [str(ch) for ch in list(self.channels.index.values)]
        xs = element.data.magnitude - element.data.magnitude.mean(axis=-1,
                                                                  keepdims=True)
        tois = []
        for c in tqdm(range(0, element.num_trials, NUM_WORKERS)):
            trials = slice(c, c + NUM_WORKERS)
            trial_xs = mne.EpochsArray(
                np.moveaxis(xs[:, :, trials], -1, 0),
                mne.create_info(channels, int(self.f0.item())), proj=False,
            )

            data = spy.mne_epochs_to_tldata(trial_xs)
            cfg = spy.get_defaults(spy.freqanalysis)
            cfg.foi = self.freqs.magnitude.squeeze()
            cfg.ft_compat = True
            cfg.keeptrials = 'yes'
            cfg.method = 'mtmconvol'
            cfg.output = 'pow'
            cfg.polyremoval = 0
            cfg.pad = 'nextpow2'
            # Temporal resolution trades off against frequency resolution.
            cfg.t_ftimwin = self._time_window
            cfg.taper = self._taper
            cfg.tapsmofrq = 4
            cfg.toi = 0.95
            tfr = spy.freqanalysis(cfg, data)
            tois.append(tfr.time[0])
            path, ext = os.path.splitext(tfr.filename)
            if self.path:
                path = self.path
            tfr.save(filename=path + "/tfr_" + str(c) + ext)
            tfr._close()

            element_data.append(tfr.filename)

            del data
            del trial_xs
            del tfr
            spy.cleanup(interactive=False)

        toi = np.array(tois).mean(axis=0) * element.times[0].units +\
              element.times[0] + self._time_window * pq.second
        self._k += 1
        if self.data is None:
            return (element_data, toi)
        else:
            return (self.data[0] + element_data, self.data[1] + toi)

    def closest_freq(self, f):
        return np.nanargmin(np.abs(self.freqs - f))

    @property
    def df(self):
        return self._df

    @property
    def dt(self):
        return (1. / self.f0).rescale('s')

    def fmap(self, f):
        return self.__class__(self.df, self.channels, self.f0, fmax=self.fmax,
                              data=f(self.data))

    @property
    def f0(self):
        return self._f0

    @property
    def fmax(self):
        return self._freqs[-1]

    @property
    def freqs(self):
        if self.data:
            return spy.load(self.data[0][0]).freq
        return self._freqs

    def result(self):
        elements = [spy.load(element) for element in self.data[0]]
        times = elements[0].sampleinfo[:, 1] - elements[0].sampleinfo[:, 0]
        shape = [len(self.channels), int(times.mean()), len(elements[0].freq)]
        tfrs = []
        ntrials = 0
        for element in elements:
            element_tfrs = element.show()
            if isinstance(element_tfrs, list):
                element_tfrs = np.stack(element_tfrs, axis=-1)
            else:
                element_tfrs = element_tfrs[:, :, :, np.newaxis]
            element_tfrs = np.moveaxis(element_tfrs, 2, 0)
            assert len(element_tfrs.shape) == 4
            tfrs.append(dask.array.from_array(element_tfrs))
            ntrials += element_tfrs.shape[-1]
            del element_tfrs
        del elements
        tfrs = dask.array.concatenate(tfrs, axis=-1)

        pows = tfrs.compute() * pq.mV ** 2 / pq.Hz
        return signals.tfr.EpochedTfr(self.channels, pows, self.df,
                                      np.diff(self.times).mean(), self.f0,
                                      self.freqs, self.times)

    @property
    def path(self):
        return self._path

    @property
    def times(self):
        return self.data[1] / self._k
