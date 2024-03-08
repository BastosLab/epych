#!/usr/bin/python3

from collections.abc import Iterable
import copy
import hdf5storage as mat
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import typing
from typing import Generic, Optional, TypeVar

from . import signal, recording

T = TypeVar('T', bound=signal.Signal)

class Statistic(Generic[T]):
    def __init__(self, iid_shape, data: Optional[np.ndarray]=None):
        assert data is None or data.shape[:len(iid_shape)] == iid_shape
        self._iid_shape = iid_shape
        self._data = data

    def apply(self, element: tuple[T, ...]) -> np.ndarray:
        raise NotImplementedError

    def calculate(self, elements: Iterable[tuple[T, ...]]):
        for element in elements:
            self._data = self.apply(element)
        return self._data

    @property
    def data(self):
        return self._data

    def fmap(self, f):
        return self.__class__(self.iid_shape, f(self.data))

    @property
    def iid_shape(self):
        return self._iid_shape

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        arrays = {k: v for k, v in self.__dict__.items()
                  if isinstance(v, np.ndarray)}
        mat.savemat(path + ("/%s.mat" % self.__class__.name), arrays)
        other = copy.copy(self)
        for k in arrays:
            setattr(other, k, None)
        with open(path + ("/%s.pickle" % self.__class__.name), mode="wb") as f:
            pickle.dump(other, f)

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + ("/%s.pickle" % self.__class__.name), mode="rb") as f:
            self = pickle.load(f)

        arrays = mat.loadmat(path + ("/%s.mat" % self.__class__.name))
        for k in arrays:
            setattr(self, k, arrays[k])
        return self

class ChannelwiseStatistic(Statistic[T]):
    def __init__(self, channels: pd.DataFrame, iid_shape,
                 data:Optional[np.ndarray]=None):
        assert isinstance(channels, pd.DataFrame)
        self._channels = channels

        iid_shape = (len(channels), *iid_shape)
        super().__init__(iid_shape, data=data)

    @property
    def channels(self):
        return self._channels

    def fmap(self, f):
        return self.__class__(self.channels, self.iid_shape, f(self.values))

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        self.channels.to_csv(path + "/channels.csv")
        arrays = {k: v for k, v in self.__dict__.items()
                  if isinstance(v, np.ndarray)}
        mat.savemat(path + ("/%s.mat" % self.__class__.name), arrays)
        other = copy.copy(self)
        for k in arrays:
            setattr(other, k, None)
        with open(path + ("/%s.pickle" % self.__class__.name), mode="wb") as f:
            pickle.dump(other, f)

    def select_channels(self, mask):
        channels = self.channels.loc[mask]
        return self.fmap(lambda vals: vals[mask, :])

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + ("/%s.pickle" % self.__class__.name), mode="rb") as f:
            self = pickle.load(f)

        arrays = mat.loadmat(path + ("/%s.mat" % self.__class__.name))
        for k in arrays:
            setattr(self, k, arrays[k])
        self._channels = pd.read_csv(path + "/channels.csv", index_col=0)
        self._channels["location"] = self._channels["location"].apply(eval)
        return self


class Summary:
    def __init__(self, statistic):
        self._stat = statistic
        self._stats = {}

    def calculate(self, elements: Iterable[recording.Sampling]):
        for element in elements:
            for k, v in element.signals.items():
                key = k + "/" + self.signal_key(v)
                if key not in self._stats:
                    self._stats[key] = self.stat()
                self._stats[key].calculate([v])
        return self._stats

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        for k, v in self.stats.items():
            v.pickle(path + "/" + k)
        other = copy.copy(self)
        other._stats = None
        with open(path + "/summary.pickle", mode="wb") as f:
            pickle.dump(other, f)

    def signal_key(self, sig: signal.Signal):
        raise NotImplementedError

    @property
    def stat(self):
        return self._stat

    @property
    def stats(self):
        return self._stats

    @classmethod
    def unpickle(cls, path, statistic_cls):
        assert os.path.isdir(path)

        with open(path + "/summary.pickle", mode="rb") as f:
            self = pickle.load(f)
        self._signals = {}
        ls = [entry.name for entry in os.scandir(path) if entry.is_dir()]
        for entry in sorted(ls):
            self._signals[entry] =\
                signal.EpochedSignal.unpickle(path + "/" + entry)
        self._statistic = statistic_cls.unpickle(
            path + "/" + statistic_cls.__name__
        )
        return self
