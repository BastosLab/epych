#!/usr/bin/python3

from collections.abc import Iterable
import numpy as np
import os
import pandas as pd
import pickle
import re

from .. import signal, statistic

class LaminarAlignment(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, column="location", data=None):
        self._column = column
        self._num_times = None
        super().__init__((1,), data=data)

    def align(self, i: int, sig: signal.EpochedSignal) -> signal.EpochedSignal:
        low, l4, high = self.result()[i]
        alignment_mask = [c in range(int(low), int(high)) for c
                          in range(len(sig.channels))]
        result = sig.select_channels(alignment_mask)
        return result.__class__(result.channels,
                                result.data[:, :self.num_times], result.dt,
                                result.times[:self.num_times])

    def apply(self, element: signal.EpochedSignal):
        area_l4 = os.path.commonprefix([l.decode() for l
                                        in element.channels.location]) + "4"
        l4_mask = [area_l4 in loc.decode() for loc in element.channels.location]

        channels_index = element.channels.channel\
                         if "channel" in element.channels.columns\
                         else element.channels.index
        l4_center = round(np.median(channels_index[l4_mask]))

        sample = np.array((channels_index.values[0], l4_center,
                           channels_index.values[-1]))[np.newaxis, :]
        if self.num_times is None or len(element) < self.num_times:
            self._num_times = len(element)
        if self.data is None:
            return sample
        return np.concatenate((self.data, sample), axis=0)

    def fmap(self, f):
        return self.__class__(self._area, self._column, f(self.data))

    @property
    def num_channels(self):
        low, _, high = self.result()[0]
        return high - low

    @property
    def num_times(self):
        return self._num_times

    def result(self):
        l4_channels = self._data[:, 1]
        low_distance = (l4_channels - self._data[:, 0]).mean()
        high_distance = (self._data[:, 2] - l4_channels).mean()
        return np.array([l4_channels - low_distance, l4_channels,
                         l4_channels + high_distance]).T.round()

def laminar_alignment(sig):
    return LaminarAlignment()

class AlignmentSummary(statistic.Summary):
    def __init__(self):
        super().__init__(laminar_alignment)

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + "/summary.pickle", mode="rb") as f:
            self = pickle.load(f)
        self._stats = {}
        ls = [entry.name for entry in os.scandir(path) if entry.is_dir()]
        for entry in sorted(ls):
            entry_ls = [area.name for area in os.scandir(path + "/" + entry)
                        if area.is_dir()]
            for area in entry_ls:
                self._stats[entry + "/" + area] =\
                    LaminarAlignment.unpickle(path + "/" + entry + "/" + area)
        self._statistic = LaminarAlignment
        return self
