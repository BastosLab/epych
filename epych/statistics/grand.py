#!/usr/bin/python3

import copy
import functools
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import cv2 as cv
import quantities as pq
import scipy
from typing import TypeVar

from .. import plotting, signal, statistic

from . import alignment

NUM_WORKERS = os.cpu_count() * 3 // 4
T = TypeVar('T', bound=signal.EpochedSignal)

class GrandConcatenation(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.LaminarAlignment, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._dt = None
        if data is None:
            self._data = {"channels": None, "k": 0, "cat": [],
                          "timestamps": np.zeros(self.iid_shape[1])}
        self._signal_class = None

    @property
    def alignment(self):
        return self._alignment

    def apply(self, element: T):
        element = self.alignment.align(self.data["k"], element)
        assert len(element.channels) == self.num_channels
        assert element.data.shape[0] == self.num_channels
        data = element.data
        if self.num_times < element.data.shape[1]:
            data = data[:, :self.num_times, :]
        running = {
            "cat": self.data["cat"],
            "channels": copy.deepcopy(self.data["channels"]),
            "k": self.data["k"],
            "timestamps": self.data["timestamps"]
        }

        channels = element.channels.reset_index(drop=True)
        if running["channels"] is None:
            running["channels"] = channels
            self._dt = element._dt
            self._signal_class = element.__class__
        else:
            for column in running["channels"].columns:
                if running["channels"][column].values.dtype == np.int64:
                    running["channels"][column] += channels[column]
        assert hasattr(self._dt, "units")

        running["k"] += 1
        if not running["cat"]:
            running["cat"] = [data]
        else:
            running["cat"].append(data.rescale(running["cat"][-1].units))
        times = element.times[:self.num_times]
        if not hasattr(running["timestamps"], "units"):
            running["timestamps"] = pq.Quantity(running["timestamps"],
                                                times.units)
        else:
            times = times.rescale(running["timestamps"].units)
        running["timestamps"] += times
        return running

    @property
    def num_channels(self):
        return self.iid_shape[0]

    @property
    def num_times(self):
        return self.iid_shape[1]

    def result(self):
        times = self.data["timestamps"] / self.data["k"]
        channels = self.data["channels"].copy()
        for column in channels.columns:
            if channels[column].values.dtype == np.int64:
                channels[column] //= self.data["k"]
        cat = pq.Quantity(
            np.concatenate(tuple(data.magnitude for data in self.data["cat"]),
                           axis=-1),
            self.data["cat"][-1].units
        )
        return self._signal_class(channels, cat, self._dt, times)

class GrandAverage(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.LaminarAlignment, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._dt = None
        if data is None:
            self._data = {"channels": None, "k": 0, "n": 0,
                          "sum": None,
                          "timestamps": np.zeros(self.iid_shape[1])}
        self._signal_class = None

    @property
    def alignment(self):
        return self._alignment

    def apply(self, element: T):
        element = self.alignment.align(self.data["k"], element)
        assert len(element.channels) == self.num_channels
        assert element.data.shape[0] == self.num_channels
        data = element.data
        if self.num_times < element.data.shape[1]:
            data = data[:, :self.num_times, :]
        running = copy.deepcopy(self.data)

        channels = element.channels.reset_index(drop=True)
        if running["channels"] is None:
            running["channels"] = channels
            self._dt = element._dt
            self._signal_class = element.__class__
        else:
            for column in running["channels"].columns:
                if running["channels"][column].values.dtype == np.int64:
                    running["channels"][column] += channels[column]

        running["k"] += 1
        running["n"] += element.num_trials

        if running["sum"] is None:
            running["sum"] = data.magnitude.sum(axis=-1, keepdims=True) * data.units
            running["timestamps"] = element.times[:self.num_times]
        else:
            running["sum"] += data.magnitude.sum(axis=-1, keepdims=True) * data.units
            running["timestamps"] += element.times[:self.num_times]
        return running

    def heatmap(self, ax=None, fig=None, title=None, vmin=None, vmax=None,
                origin="lower"):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        data = self.result().squeeze()
        plotting.heatmap(fig, ax, data, cbar=False, title=title, vmin=vmin,
                         vmax=vmax)

        num_xticks = len(ax.get_xticks())
        xtick_locs = np.linspace(0, data.shape[1], num_xticks)
        xticks = np.linspace(0, data.shape[-1], num_xticks)
        xticks = ["%0.2f" % t for t in xticks]
        ax.set_xticks(xtick_locs, xticks)

    @property
    def num_channels(self):
        return self.iid_shape[0]

    @property
    def num_times(self):
        return self.iid_shape[1]

    def plot(self, **kwargs):
        return self.result().plot(**kwargs)

    def result(self):
        data = self.data["sum"] / self.data["n"]
        times = self.data["timestamps"] / self.data["k"]
        channels = self.data["channels"].copy()
        for column in channels.columns:
            if channels[column].values.dtype == np.int64:
                channels[column] //= self.data["k"]
        return self._signal_class(channels, data, self._dt, times).evoked()

class GrandVariance(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.ChannelAlignment,
                 mean: signal.EvokedSignal, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._mean = mean
        if data is None:
            self._data = {"diffs": None, "k": 0, "n": 0}

    @property
    def alignment(self):
        return self._alignment

    def apply(self, element: T):
        element = self.alignment.align(self.data["k"], element)
        assert len(element.channels) == self.alignment.num_channels
        assert element.data.shape[0] == self.alignment.num_channels
        data = element.data
        if self.alignment.num_times < element.data.shape[1]:
            data = data[:, :self.alignment.num_times, :]
        running = copy.deepcopy(self.data)

        if running["diffs"] is None:
            running["diffs"] = ((data - self.mean.data) ** 2).magnitude.sum(
                axis=-1, keepdims=True
            ) * self.mean.data.units
        else:
            running["diffs"] += ((data - self.mean.data) ** 2).magnitude.sum(
                axis=-1, keepdims=True
            ) * self.mean.data.units
        running["k"] += 1
        running["n"] += element.num_trials
        return running

    @property
    def mean(self):
        return self._mean

    def result(self):
        variance = self.data["diffs"] / (self.data["n"] - 1)
        return self.mean.__class__(self.mean.channels, variance, self.mean._dt,
                                   self.mean.times)

class GrandNonparametricClusterTest(statistic.Statistic[T]):
    def __init__(self, num_spatial_channels, num_times, alpha=0.05, data=None,
                 partitions=1000):
        super().__init__((num_spatial_channels, num_times), data=data)
        self._alpha = alpha
        if data is None:
            self._data = {"left": None, "right": None}
        self._partitions = partitions
        self._result = {}

    @property
    def alpha(self):
        return self._alpha

    def apply(self, element: tuple[T, T]):
        assert self.data["left"] is None and self.data["right"] is None
        assert np.isclose(element[0].dt.magnitude, element[1].dt.magnitude)
        assert (element[0].channels == element[1].channels).all().all()

        return {"left": element[0], "right": element[1]}

    @property
    def partitions(self):
        return self._partitions

    def plot(self, fmask=None, fsig=None, **kwargs):
        contrast = self.result()
        if fmask is None:
            mask = contrast["mask"]
        else:
            mask = fmask(contrast["mask"])
        if fsig is None:
            signal = contrast["signal"]
        else:
            signal = fsig(contrast["signal"])

        signal = signal.fmap(lambda data: data * mask[:, :, np.newaxis])
        return signal.plot(**kwargs)

    def result(self):
        if not self._result:
            lunits = self.data["left"].data.units
            ldata = self.data["left"].data.magnitude
            rdata = self.data["right"].data.rescale(lunits).magnitude

            threshold = scipy.stats.f.ppf(
                1 - self.alpha, dfn=2 - 1,
                dfd=ldata.shape[-1] + rdata.shape[-1] - 1
            )
            Fs, clusters, pvals, H0s = mne.stats.spatio_temporal_cluster_test(
                (np.swapaxes(ldata, 0, -1), np.swapaxes(rdata, 0, -1)),
                n_jobs=NUM_WORKERS, n_permutations=self.partitions,
                out_type="mask", tail=1, threshold=threshold
            )
            cluster_masks = [clusters[c] for c
                             in np.where(pvals < self.alpha)[0]]
            mask_shape = ldata.shape[:-1]
            null_mask = np.resize(np.array([False]),
                                  mask_shape[1:] + (mask_shape[0],))
            mask = functools.reduce(np.logical_or, cluster_masks, null_mask)
            mask = np.expand_dims(np.moveaxis(mask, -1, 0), -1)
            data = (ldata.mean(axis=-1) - rdata.mean(axis=-1)) * lunits
            self._result = {
                "mask": mask, "signal": self.data["left"].__replace__(
                    data=np.expand_dims(data, -1)
                ).evoked()
            }
        return self._result

def t_stats(ln, lmean, lvar, rn, rmean, rvar):
    l_stderr, r_stderr = lvar / ln, rvar / rn
    ts = (lmean - rmean) / np.sqrt(l_stderr + r_stderr)
    dfs = (l_stderr + r_stderr) ** 2
    dfs /= l_stderr ** 2 / (ln - 1) + r_stderr ** 2 / (rn - 1)
    pvals = scipy.special.stdtr(dfs, -np.abs(ts)) * 2
    return ts, pvals

def t_test(left: GrandVariance, right: GrandVariance):
    assert left.iid_shape == right.iid_shape
    k = n = 0

    return t_stats(left.data["n"], left.mean.data.magnitude,
                   left.result().data.magnitude, right.data["n"],
                   right.mean.data.magnitude, right.result().data.magnitude)

def summary_t_test(left: statistic.Summary, right: statistic.Summary):
    results = {}
    for key in (left.stats.keys() & right.stats.keys()):
        results[key] = t_test(left.stats[key], right.stats[key])
    return results
