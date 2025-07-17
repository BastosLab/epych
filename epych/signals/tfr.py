#!/usr/bin/python3

import dask.array
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import quantities as pq

from .. import plotting, signal
from ..statistics import spectrum

class TimeFrequencyRepr(signal.Signal):
    def __init__(self, channels: pd.DataFrame, data, df, dt, f0, freqs,
                 timestamps):
        self._df = df
        self._f0 = f0
        self._freqs = freqs
        super().__init__(channels, data, dt, timestamps)

    def band_power(self, fbottom, ftop):
        raise NotImplementedError

    def baseline(self, start, end, decibels=False):
        first = np.abs(self.times - start).argmin()
        last = np.abs(self.times - end).argmin()
        base_mean = self.data[:, first:last, :].magnitude.mean(
            axis=1, keepdims=True
        )
        base_mean = base_mean * self.data.units

        if decibels and self.data.units == spectrum.decibel:
            tfrs = self.data - base_mean
        elif decibels and self.data.units != spectrum.decibel:
            tfrs = 10 * np.log10(self.data / base_mean) * spectrum.decibel
        else:
            tfrs = (self.data - base_mean) / base_mean * 100 * pq.percent
        return self.__replace__(data=tfrs)

    def channel_depths(self, column=None):
        if column is not None and column in self.channels:
            return self.channels[column].values
        return np.arange(len(self.channels))

    def channel_mean(self):
        middle_channel = len(self.channels) // 2
        channels = self.channels[middle_channel:(middle_channel + 1)]
        data = self.data.magnitude.mean(axis=0, keepdims=True) * self.data.units
        return self.__replace__(channels=channels, data=data)

    def channel_sum(self):
        middle_channel = len(self.channels) // 2
        channels = self.channels[middle_channel:(middle_channel + 1)]
        data = self.data.magnitude.sum(axis=0, keepdims=True) * self.data.units
        return self.__replace__(channels=channels, data=data)

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    def decibels(self):
        dBs = spectrum.decibel
        return self.fmap(lambda data: 10 * np.log10(data.magnitude) * dBs)

    @property
    def df(self):
        return self._df

    @property
    def f0(self):
        return self._f0

    @property
    def freqs(self):
        return self._freqs

    def fmap(self, f):
        return self.__replace__(data=f(self.data))

    @property
    def fmax(self):
        return self._freqs[-1]

    def relative(self):
        return self.__replace__(
            data=self.data / self.data.max(axis=0)[np.newaxis, :, :, :]
        )

    def __replace__(self, /, **changes):
        parameters = {field: changes.get(field, getattr(self, field)) for field
                      in ["channels", "data", "df", "dt", "f0", "freqs",
                          "times"]}
        return self.__class__(*parameters.values())

    def select_freqs(self, low, high):
        low_idx = np.argmin(np.abs(self.freqs - low))
        high_idx = np.argmin(np.abs(self.freqs - high)) + 1
        return self.__replace__(data=self.data[:, :, low_idx:high_idx, :],
                                freqs=self.freqs[low_idx:high_idx])

class EpochedTfr(TimeFrequencyRepr, signal.EpochedSignal):
    def __init__(self, channels: pd.DataFrame, data, df, dt, f0, freqs,
                 timestamps):
        assert len(data.shape) == 4
        assert len(channels) == data.shape[0]
        assert len(timestamps) == data.shape[1]
        assert timestamps.units == dt.units

        super(EpochedTfr, self).__init__(channels, data, df, dt, f0, freqs,
                                         timestamps)

    def band_power(self, fbottom, ftop):
        ibot = self.closest_freq(fbottom)
        itop = self.closest_freq(ftop)
        return signal.EpochedSignal(self.channels,
                                    self.data[:, :, ibot:itop].mean(axis=2),
                                    self.dt, self.times)

    def evoked(self):
        data = pq.Quantity(self.data.magnitude.mean(-1, keepdims=True),
                           self.data.units)
        return EvokedTfr(self.channels, data, self.df, self.dt, self.f0,
                         self.freqs, self.times)

    def oscillatory(self, channel_mean=False, mode="fixed", space="linear"):
        _, freqs, aperiodic = self.power_spectrum().oscillatory(channel_mean,
                                                                mode, space)
        return self.remove_aperiodic(aperiodic, space)

    def power_spectrum(self):
        return spectrum.PowerSpectrum(self.df, self.channels, self.f0,
                                      fmax=self.freqs[-1], freqs=self.freqs,
                                      data=self.data.mean(axis=1))

    def remove_aperiodic(self, aperiodic, space="linear"):
        aperiodic = aperiodic[:, np.newaxis, :, :]
        if space == "linear":
            return self.fmap(lambda data: data.magnitude / aperiodic *\
                                          pq.dimensionless)
        elif space == "log":
            return self.fmap(
                lambda data: 10 * (np.log10(data.magnitude) - aperiodic) *\
                             spectrum.decibel
            )
        raise NotImplementedError

    @classmethod
    def unpickle(cls, path):
        self = super(cls).unpickle(path)
        if not hasattr(self.freqs, "units"):
            self._freqs = pq.Quantity(self.freqs, pq.Hz)

class EvokedTfr(TimeFrequencyRepr, signal.EvokedSignal):
    def __init__(self, channels: pd.DataFrame, data, df, dt, f0, freqs,
                 timestamps):
        assert data.shape[-1] == 1
        super(EvokedTfr, self).__init__(channels, data, df, dt, f0, freqs,
                                        timestamps)

    def band_power(self, fbottom, ftop):
        ibot = self.closest_freq(fbottom)
        itop = self.closest_freq(ftop)
        return EvokedPower(self.channels,
                           self.data[:, :, ibot:itop].mean(axis=2), self.dt,
                           self.times)

    def evoked(self):
        erp = super().evoked()
        return EvokedTfr(erp.channels, erp.data, erp.df, erp.dt, erp.f0,
                         self.freqs, erp.times)

    def heatmap(self, alpha=None, ax=None, cmap=None, fbottom=0, fig=None,
                filename=None, ftop=None, title=None, vlim=None, vmin=None,
                vmax=None, baseline=None, cbar=True, cbar_ends=None,
                tlabel="Time (milliseconds)", contours=False, **events):
        lone = fig is None
        if fig is None:
            fig = plt.figure(figsize=(self.plot_width * 4, 3), dpi=100)
        if alpha is not None:
            alpha = alpha.squeeze()
        if ax is None:
            ax = fig.add_subplot()
        if ftop is None:
            ftop = self.fmax.item()
        vlim = self.data.max().round(decimals=1) if vlim is None else vlim
        if vlim == 0.:
            vlim = 1
        if vmax is None:
            vmax = vlim
        if vmin is None:
            vmin = -vlim

        freqs = self.freqs
        times = self.times.rescale("ms")
        tfrs = self.channel_mean().data.squeeze()
        title = "Spectrogram" if title is None else title
        if tfrs.units.dimensionality.string == "%":
            title += " (% change from baseline)"
        img = plotting.heatmap(fig, ax, tfrs.T, alpha=alpha, cmap=cmap,
                               title=title, vmin=vmin, vmax=vmax, cbar=cbar,
                               cbar_ends=cbar_ends, contours=contours)

        ax.set_xlim(0, len(times))
        xticks = ax.get_xticks().astype(int)
        zero_tick = self.sample_at(0.)
        zerotick_loc = ((xticks - zero_tick) ** 2).argmin()
        zerotick_offset = (xticks - zero_tick)[zerotick_loc]
        xticks -= zerotick_offset
        while xticks[-1] >= len(times):
            xticks = xticks[:-1]
        xtick_times = times[xticks].round(decimals=0)
        ax.set_xticks(xticks, xtick_times.magnitude.astype(int))
        ax.set_xlabel(tlabel)

        ax.set_ylim(0, tfrs.shape[-1])
        yticks = [int(ytick) for ytick in ax.get_yticks()]
        yticks[-1] = min(yticks[-1], tfrs.shape[-1] - 1)
        ax.set_yticks(yticks, ['{}'.format(int(f)) for f in freqs[yticks]])
        ymin, ymax = ax.get_ybound()
        ax.set_ylabel("Frequency (Hz)")

        if baseline is not None:
            bxmin = self.sample_at(baseline[0])
            bxmax = self.sample_at(baseline[1])
            ax.axvspan(bxmin, bxmax, alpha=0.1, color='k')

        for (event, (time, color)) in events.items():
            xtime = self.sample_at(time)
            ax.vlines(xtime, *ax.get_ybound(), colors=color,
                      linestyles='dashed', label=event)
            ax.annotate(event, (xtime + 0.5, ymax - 1), color=color)

        band_bounds = np.unique(list(spectrum.THETA_BAND) +\
                                list(spectrum.ALPHA_BETA_BAND) +\
                                list(spectrum.GAMMA_BAND)) * pq.Hz
        yfreqs = [np.nanargmin(np.abs(freqs - bound)) for bound in band_bounds]
        ax.hlines(yfreqs, *ax.get_xbound(), colors='gray', linestyles='dotted')

        if filename is not None:
            fig.savefig(filename, dpi=100, bbox_inches="tight")
        if lone:
            plt.show()
            plt.close(fig)
        return img

    def plot(self, *args, **kwargs):
        return self.heatmap(*args, **kwargs)

    @property
    def plot_width(self):
        width = (self.times[-1] - self.times[0])
        if hasattr(width, "units"):
            width = width.magnitude
        return width

    def power_spectrum(self):
        return spectrum.PowerSpectrum(self.df, self.channels, self.f0,
                                      fmax=self.freqs[-1], freqs=self.freqs,
                                      data=self.data.mean(axis=1))

    def spectrolaminar_plot(self, depth_column="vertical", filename=None,
                            title=None, xlims=[0.2, 0.5], **bands):
        if title is None:
            title = os.path.commonprefix(
                [chan.decode() if isinstance(chan, bytes) else chan for chan in
                 self.channels["location"].values]
            )
        pows = {}
        for name, (low, high, k) in bands.items():
            pows[k] = self.relative().band_power(low, high).data.magnitude
            pows[k] = pows[k].mean(axis=1).squeeze()
        totals = sum(list(pows.values()))
        pows = {k: v / totals for k, v in pows.items()}
        depths = self.channel_depths(column=depth_column)

        fig = plt.figure(figsize=(3, (depths[-1] - depths[0]) / 1000 * 4))
        ax = fig.add_subplot()
        power_lines = list(itertools.chain(*[[pows[k], depths]
                                           for (k, v) in pows.items()]))
        ax.plot(*power_lines)
        ax.set_xlabel("Relative spectral power (out of 1.0)")
        if xlims is not None:
            ax.set_xlim(xlims)
        ax.legend(list(pows.keys()))

        ax.set_title(title)
        self.annotate_channels(ax, "location", ycolumn="vertical")

        if filename is not None:
            fig.savefig(filename, dpi=100)
        plt.show()
        plt.close(fig)

    @classmethod
    def unpickle(cls, path):
        self = super(__class__, cls).unpickle(path)
        if not hasattr(self.freqs, "units"):
            self._freqs = pq.Quantity(self.freqs, pq.Hz)
        return self

class EvokedPower(signal.EvokedSignal):
    def heatmap(self, alpha=None, ax=None, cmap=None, fig=None,
                filename=None, ftop=None, title=None, vlim=None, vmin=None,
                vmax=None, baseline=None, cbar=True, cbar_ends=None,
                tlabel="Time (milliseconds)", channel_ticks="location",
                width=4, contours=False, **events):
        lone = fig is None
        if fig is None:
            fig = plt.figure(figsize=(width, 3), dpi=100)
        if alpha is not None:
            alpha = alpha.squeeze()
        if ax is None:
            ax = fig.add_subplot()
        vlim = 0.8 * self.data.max() if vlim is None else vlim
        if vlim == 0.:
            vlim = 1
        if vmax is None:
            vmax = vlim
        if vmin is None:
            vmin = -vlim

        times = self.times.rescale("ms")
        img = plotting.heatmap(fig, ax, self.data.squeeze(), alpha=alpha,
                               cmap=cmap, title=title, vmin=vmin, vmax=vmax,
                               cbar=cbar, cbar_ends=cbar_ends,
                               contours=contours)

        ax.set_xlim(0, len(times))
        xticks = ax.get_xticks().astype(int)
        zero_tick = self.sample_at(0.)
        zerotick_loc = ((xticks - zero_tick) ** 2).argmin()
        zerotick_offset = (xticks - zero_tick)[zerotick_loc]
        xticks -= zerotick_offset
        if xticks[-1] >= len(times):
            xticks = xticks[:-1]
        xtick_times = times[xticks].round(decimals=0)
        ax.set_xticks(xticks, xtick_times.magnitude.astype(int))
        ax.set_xlabel(tlabel)

        if channel_ticks is not None and channel_ticks in self.channels.columns:
            self.annotate_channels(ax, channel_ticks)

        if baseline is not None:
            bxmin = self.sample_at(baseline[0])
            bxmax = self.sample_at(baseline[1])
            ax.axvspan(bxmin, bxmax, alpha=0.1, color='k')

        for (event, (time, color)) in events.items():
            xtime = self.sample_at(time)
            ax.vlines(xtime, *ax.get_ybound(), colors=color,
                      linestyles='dashed', label=event)
            ax.annotate(event, (xtime + 0.5, ax.get_ybound()[1] - 1),
                        color=color)

        if filename is not None:
            fig.savefig(filename, dpi=100, bbox_inches="tight")
        if lone:
            plt.show()
            plt.close(fig)
        return img

    def plot(self, *args, **kwargs):
        return self.heatmap(*args, **kwargs)

class BandPower(signal.Signal):
    def __init__(self, bands: pd.DataFrame, data, dt, timestamps):
        super().__init__(bands, data, dt, timestamps)

    def __replace__(self, /, **changes):
        parameters = {field: changes.get(field, getattr(self, field)) for field
                      in ["bands", "channels", "data", "dt", "times"]}
        return self.__class__(*parameters.values())

class EpochedBandPower(BandPower, signal.EpochedSignal):
    def __init__(self, bands: pd.DataFrame, data, dt, timestamps):
        assert len(data.shape) == 4
        assert len(bands) == data.shape[0]
        assert len(timestamps) == data.shape[1]
        assert timestamps.units == dt.units

        super(EpochedBandPower, self).__init__(bands, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedBandPower(erp.channels, erp.data, erp.dt, erp.times)

class EvokedBandPower(BandPower, signal.EvokedSignal):
    def __init__(self, bands: pd.DataFrame, data, dt, timestamps):
        assert data.shape[-1] == 1
        super(EvokedBandPower, self).__init__(bands, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedBandPower(erp.channels, erp.data, erp.dt, erp.times)

    def plot(self, *args, **kwargs):
        return self.line_plot(*args, **kwargs)

    @property
    def plot_width(self):
        width = (self.times[-1] - self.times[0])
        if hasattr(width, "units"):
            width = width.magnitude
        return width
