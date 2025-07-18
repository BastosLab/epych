#!/usr/bin/python3

from collections import Counter
import collections.abc as abc
import copy
import functools
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import quantities as pq
import typing

from . import preprocess, signal

def empty_intervals():
    return pd.DataFrame(columns=["trial", "type", "start", "end"])

def empty_trials():
    return pd.DataFrame(columns=["trial"]).set_index("trial")

def default_units(time_unit=pq.second):
    return {"start": pq.second, "end": pq.second}

class Sampling(abc.Sequence):
    def __init__(self, intervals: pd.DataFrame, trials: pd.DataFrame,
                 units: dict[str, pq.UnitQuantity], **signals):
        for column in units:
            assert column in intervals.columns or column in trials.columns
        assert set(intervals.columns) >= {"type", "start", "end"}
        assert trials.index.name == "trial"
        assert len(trials) == 0 or any(signal.num_trials == len(trials) for
                                       signal in signals.values())
        assert isinstance(units["start"], pq.UnitTime) and\
               isinstance(units["end"], pq.UnitTime)
        self._intervals = intervals
        self._signals = signals
        self._trials = trials
        self._units = units

    def __add__(self, other):
        assert self.signals.keys() == other.signals.keys()
        trials = self.trials.merge(
            other.trials,
            on=list(set(self.trials.columns) & set(other.trials.columns)) + ["trial"]
        )
        assert self.units == other.units
        intervals = empty_intervals()
        signals = {k: self.signals[k] + other.signals[k] for k in self.signals}
        return self.__class__(intervals, trials, self.units, **signals)

    def baseline_correct(self, start, stop):
        return self.smap(lambda v: v.baseline_correct(start, stop))

    def cat_trials(self, other):
        assert self.signals.keys() == other.signals.keys()
        trials = self.trials.merge(
            other.trials,
            on=list(set(self.trials.columns) & set(other.trials.columns)) + ["trial"]
        )
        assert self.units == other.units
        intervals = empty_intervals()
        signals = {k: v.cat_trials(other.signals[k]) for k, v
                   in self.signals.items()}
        return self.__class__(intervals, trials, self.units, **signals)

    def erp(self):
        intervals = []
        for epoch_type in self.intervals["type"].unique():
            epochs = self.intervals.loc[self.intervals["type"] == epoch_type]
            epochs = epochs.mean(0, numeric_only=True)
            epochs = pd.DataFrame(data=epochs.values[np.newaxis, :],
                                  columns=epochs.index)
            intervals.append(epochs.assign(type=[epoch_type]))
        if intervals:
            intervals = pd.concat(intervals)
        else:
            intervals = empty_intervals()

        trials = self.trials.mean(axis=0, numeric_only=True)
        trials = pd.DataFrame(data=trials.values[np.newaxis, :],
                              columns=trials.index.values)
        trials = trials.assign(trial=[0]).set_index("trial")

        signals = {k: v.evoked() for k, v in self.signals.items()}
        return EvokedSampling(intervals, trials, self.units, **signals)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if key.step is None:
            key = slice(key.start, key.stop, 1)

        intervals = np.repeat([key.start, key.stop], len(self.trials), axis=-1)
        return self.time_lock(intervals)

    @property
    def intervals(self):
        return self._intervals

    def __len__(self):
        return min(len(signal) for signal in self.signals.values())

    @property
    def num_trials(self):
        return min(signal.num_trials for signal in self.signals.values())

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        self.intervals.to_csv(path + "/intervals.csv")
        self.trials.to_csv(path + "/trials.csv")
        for k, v in self.signals.items():
            v.pickle(path + "/" + k)
        other = copy.copy(self)
        other._intervals = other._signals = other._trials = None
        with open(path + "/sampling.pickle", mode="wb") as f:
            pickle.dump(other, f)

    def select_trials(self, selections):
        extended_selections = selections + [False] *\
                              (len(self.trials) - len(selections))
        trials = self.trials.loc[extended_selections]
        signals = {k: s.select_trials(selections) for k, s
                   in self.signals.items()}
        return Sampling(self.intervals, trials, self.units, **signals)

    @property
    def signals(self):
        return self._signals

    def smap(self, f, keys=False):
        if keys:
            signals = {
                k: f(k, v) for k, v in self.signals.items()
            }
        else:
            signals = {
                k: f(v) for k, v in self.signals.items()
            }
        return self.__class__(self.intervals, self.trials, self.units,
                              **signals)

    def __sub__(self, other):
        assert self.signals.keys() == other.signals.keys()
        trials = self.trials.merge(
            other.trials,
            on=list(set(self.trials.columns) & set(other.trials.columns)) + ["trial"]
        )
        assert self.units == other.units
        intervals = empty_intervals()
        signals = {k: self.signals[k] - other.signals[k] for k in self.signals}
        return self.__class__(intervals, trials, self.units, **signals)

    def time_lock(self, times, before=0., after=0.):
        if isinstance(times, float):
            times = np.ones(len(self.trials)) * times
        onsets, offsets = times - before, times + after
        inner_intervals = self.intervals["start"].values >= onsets.mean() &\
                          self.intervals["end"].values <= offsets.mean()
        inner_intervals = self.intervals.loc[inner_intervals]
        intervals = np.stack((onsets, offsets), axis=-1)
        return self.__class__(inner_intervals, self.trials, self.units,
                              **{k: v.epoch(intervals, -before) for k, v
                              in self.signals.items()})

    @property
    def trials(self):
        return self._trials

    @property
    def units(self):
        return self._units

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + "/sampling.pickle", mode="rb") as f:
            self = pickle.load(f)
        self._signals = {}
        ls = [entry.name for entry in os.scandir(path) if entry.is_dir()]
        for entry in sorted(ls):
            with open(path + "/" + entry + "/epoched_signal.pickle",
                      mode="rb") as f:
                self._signals[entry] = pickle.load(f)
            self._signals[entry] = \
                self._signals[entry].__class__.unpickle(path + "/" + entry)
        self._trials = pd.read_csv(path + "/trials.csv", index_col="trial")
        self._intervals = pd.read_csv(path + "/intervals.csv")
        return self

class RawRecording(Sampling):
    def __init__(self, intervals: pd.DataFrame, trials: pd.DataFrame,
                 units: dict[str, pq.UnitQuantity], **signals):
        assert len(trials) <= 1
        for k, v in signals.items():
            assert isinstance(v, signal.RawSignal)
        super().__init__(intervals, trials, units, **signals)

    def epoch(self, inner_epochs, outer_epochs=None, before=0., after=0.):
        assert hasattr(after, "units")
        assert hasattr(before, "units")
        after = after.rescale(self.units["start"])
        before = before.rescale(self.units["end"])

        targets = self.intervals.loc[inner_epochs]
        if outer_epochs is not None:
            parent = self.intervals.loc[outer_epochs]

            mask = []
            target_times = zip(targets["start"], targets["end"])
            for t, (start, end) in enumerate(target_times):
                idx = np.where((parent["start"] < start) & (end < parent["end"]))[0]
                if len(idx):
                    mask.append((targets.index[t], idx[0]))
            mask = np.array(mask)
            targets, epochs = targets.loc[mask[:, 0]], parent.loc[mask[:, 1]]
        else:
            epochs = targets
        assert len(targets) == len(epochs)
        befores = (targets["start"].values - epochs["start"].values).mean()
        befores = befores * self.units["start"] + before
        afters = (epochs["end"].values - targets["end"].values).mean()
        afters = afters * self.units["end"] + after
        onsets, offsets = targets["start"] - befores, targets["end"] + afters

        epoch_intervals = np.stack((onsets.values, offsets.values), axis=-1)
        trials = []
        for t, (onset, offset) in enumerate(epoch_intervals):
            inners = (self.intervals["start"] > onset) &\
                     (self.intervals["end"] < offset)
            inners = self.intervals.loc[inners]
            inners = inners.assign(trial=[t] * len(inners))
            inners.loc[:, "start":"end"] -= onset
            for inner in inners.itertuples():
                remainder = {
                    k: v for k, v in inner._asdict().items() if k not in
                    {"trial", "type", "start", "end", "Index"}
                }
                trials.append({
                    "trial": t,
                    inner.type + "_start": inner.start - befores.magnitude,
                    inner.type + "_end": inner.end - befores.magnitude,
                    **remainder
                })
        trial_columns = [set(trial.keys()) for trial in trials]
        trial_columns = trial_columns[0].union(*trial_columns[1:])
        trials = {
            k: [trial[k] if (k in trial) else pd.NA for trial in trials]
            for k in trial_columns
        }
        trials = pd.DataFrame(data=trials,
                              columns=list(trial_columns)).set_index("trial")
        trials = trials.groupby("trial").sum()
        signals = {k: s.epoch(epoch_intervals, -befores) for k, s in
                   self.signals.items()}
        return Sampling(empty_intervals(), trials, self.units, **signals)

class EvokedSampling(Sampling):
    def __init__(self, intervals: pd.DataFrame, trials: pd.DataFrame,
                 units: dict[str, pq.UnitQuantity], **signals):
        assert len(trials) <= 1
        for k, v in signals.items():
            assert isinstance(v, signal.EvokedSignal)
        super().__init__(intervals, trials, units, **signals)

    def plot(self, alphas={}, vmin=None, vmax=None, dpi=100, figure=None,
             figargs={}, sigtitle=None, cmap=None, signals=None, title=None,
             baseline=None, decimals=1, width_per_signal=None, contours=False,
             xlabel=None, **events):
        if signals is None:
            signals = list(self.signals.keys())
        if vmin is None and vmax is None:
            vlims = np.array([
                sig.data.magnitude.max().round(decimals=decimals)
                for k, sig in self.signals.items() if k in signals
            ])
            vmaxes = np.nonzero(vlims)[0]
            vlims = np.max(vlims[vmaxes]) if len(vmaxes) else None
            vmin, vmax = (-vlims, vlims) if vlims else (None, None)
        if width_per_signal is None:
            width_per_signal = 4. * pq.point
        plot_width = int(len(signals) * width_per_signal)
        fig, axes = plt.subplot_mosaic([signals], figsize=(plot_width, 3),
                                       dpi=dpi, layout="compressed")

        for sig, ax in axes.items():
            name = sig
            if sigtitle is not None:
                name = sigtitle(sig, self.signals[sig])
            alpha = alphas.get(sig, None)
            img = self.signals[sig].plot(alpha=alpha, ax=ax, fig=fig,
                                         title=name, baseline=baseline,
                                         vmin=vmin, vmax=vmax, cbar=False,
                                         cmap=cmap, contours=contours,
                                         xlabel=xlabel)
            for (event, (time, color)) in events.items():
                ymin, ymax = ax.get_ybound()
                xtime = self.signals[sig].sample_at(time)
                ax.vlines(xtime, ymin, ymax, colors=color, linestyles='dashed',
                          label=event)
                ax.annotate(event, (xtime + 0.5, ymax - 2))

        if hasattr(self.signals[sig].data, "units"):
            label = self.signals[sig].data.units.dimensionality.latex
            if "%" in label:
                label = label.replace("%", r"\%")
        else:
            label = None
        cbar = fig.colorbar(img, ax=list(axes.values()), pad=0.01, label=label)

        if title is not None:
            fig.suptitle(title, fontsize=16)
        if figure is not None:
            fig.savefig(figure, **figargs)
        plt.show()
        plt.close(fig)

    def plot_signal(self, name, alpha=None, vmin=None, vmax=None, path=None,
                    figargs={}, sigtitle=None, cmap=None, **events):
        timespan = self.signals[name].times[-1] - self.signals[name].times[0]
        timespan *= 4
        if hasattr(timespan, "units"):
            timespan = timespan.magnitude
        fig = plt.figure(figsize=(timespan, 3))
        ax = fig.subplots()

        title = sigtitle(name, self.signals[name]) if sigtitle is not None\
                else name
        self.signals[name].plot(alpha=alpha, ax=ax, fig=fig, title=title,
                                vmin=vmin, vmax=vmax, cmap=cmap)
        for (event, (time, color)) in events.items():
            ymin, ymax = ax.get_ybound()
            xtime = self.signals[name].sample_at(time)
            ax.vlines(xtime, ymin, ymax, colors=color,
                      linestyles='dashed', label=event)
            ax.annotate(event, (xtime + 0.01, ymax))

        plt.show()
        if path is not None:
            path = path + "/" + name
            fig.savefig(path + ".pdf", bbox_inches='tight', **figargs)
            fig.savefig(path + ".png", bbox_inches='tight', **figargs)
            fig.savefig(path + ".svg", bbox_inches='tight', **figargs)
        plt.close(fig)

    def plot_signals(self, path, alphas={}, vmins={}, vmaxs={}, figargs={},
                     sigtitle=None, cmap=None, **events):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        for sig in self.signals:
            self.plot_signal(sig, alpha=alphas.get(sig, None),
                             vmin=vmins.get(sig, None), path=path,
                             vmax=vmaxs.get(sig, None), figargs=figargs,
                             sigtitle=sigtitle, cmap=cmap, **events)

def trials_ttest(sa: Sampling, sb: Sampling, pvalue=0.05):
    assert isinstance(sa, Sampling)
    assert sa.__class__ == sb.__class__
    assert sa.signals.keys() == sb.signals.keys()
    trials = sa.trials.merge(
        sb.trials,
        on=list(set(sa.trials.columns) & set(sb.trials.columns)) + ["trial"]
    )
    assert sa.units == sb.units
    intervals = empty_intervals()
    signals = {
        k: signal.trials_ttest(sa.signals[k], sb.signals[k], pvalue=pvalue)
        for k in sa.signals
    }
    return sa.__class__(intervals, trials, sa.units, **signals)
