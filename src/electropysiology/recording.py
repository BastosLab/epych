#!/usr/bin/python3

import numpy as np

class ConditionTrials:
    def __init__(self, dt, event_codes, event_times, lfp=None, mua=None,
                 spikes=None, zscore_mua=True):
        assert lfp or mua or spikes
        self._dt = dt
        self._event_codes = event_codes
        self._lfp, self._mua, self._spikes = lfp, mua, spikes

        self._ntimes = None
        for thing in (lfp, mua, spikes):
            if thing:
                assert len(thing.shape) == 3 # Channels x Times x Trials
                if ntimes:
                    assert thing.shape[1] == self._ntimes
                else:
                    self._ntimes = thing.shape[1]

    @property
    def dt(self):
        return self._dt

    @property
    def f0(self):
        return 1. / self.dt

    @property
    def fNQ(self):
        return self.f0 / 2.

    @property
    def T(self):
        return self.dt * self._ntimes
