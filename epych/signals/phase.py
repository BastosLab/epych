#!/usr/bin/python3

from elephant.parallel import ProcessPoolExecutor
from elephant.current_source_density import estimate_csd
import matplotlib.pyplot as plt
from neo import AnalogSignal
import numpy as np
import pandas as pd
import quantities as pq

from .. import signal
from ..statistics.spectrum import PowerSpectrum

class OscillationPhase(signal.Signal):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert data.units == pq.rad
        super().__init__(channels, data, dt, timestamps)

    def channel_depths(self, column=None):
        if column is not None and column in self.channels:
            return self.channels[column].values
        return np.arange(len(self.channels))

class EpochedPhase(OscillationPhase, signal.EpochedSignal):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert data.units == pq.rad
        super().__init__(channels, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedPhase(erp.channels, erp.data, erp.dt, erp.times)

class EvokedPhase(OscillationPhase, signal.EvokedSignal):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert data.shape[2] == 1
        assert data.units == pq.rad
        super().__init__(channels, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedPhase(erp.channels, erp.data, erp.dt, erp.times)
