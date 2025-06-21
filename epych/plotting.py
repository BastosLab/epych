#!/usr/bin/python3

import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from . import colormaps

def plot_sem(x, ys, ax=None, axis=-1, sems=2, **kwargs):

    if ax is None:
        ax = plt.gca()
    mean = ys.mean(axis=axis)
    sem = ys.std(axis=axis) / ys.shape[axis]
    lines = ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean - sems * sem, mean + sems * sem,
                    color=mcolors.to_rgba(lines[0].get_color(), 0.25))

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def imagesc(ax, cs, alpha=None, cmap=None, smooth=True, **kwargs):
    if cmap is None:
        cmap = colormaps.parula
    if kwargs['vmin'] is None and kwargs['vmax'] is None:
        std_dev = cs.std()
        kwargs['vmin'] = 2 * -std_dev
        kwargs['vmax'] = 2 * std_dev
    x, y = np.linspace(0, cs.shape[1]), np.linspace(0, cs.shape[0])

    kernel = np.ones((4, 4))
    kernel /= math.prod(kernel.shape)
    cs = cv.filter2D(cs, -1, kernel)

    return ax.imshow(cs, alpha=alpha, aspect='auto', interpolation='none',
                     extent=extents(x) + extents(y), cmap=cmap, **kwargs)

def heatmap(fig, ax, data, alpha=None, title=None, cbar=True, vmin=-1e-4,
            vmax=1e-4, cmap=None, smooth=True, cbar_ends=None):
    cbar = cbar or vmin is None or vmax is None
    img = imagesc(ax, data, alpha=alpha, vmin=vmin, vmax=vmax, origin='lower',
                  cmap=cmap, smooth=smooth)
    if cbar:
        if hasattr(data, "units"):
            label = data.units.dimensionality.latex
            if "%" in label:
                label = label.replace("%", r"\%")
        else:
            label = None

        cbar = fig.colorbar(img, ax=ax, pad=0.01, label=label)
        if cbar_ends:
            yticks = cbar.ax.get_yticks()
            yticklabels = cbar.ax.get_yticklabels()
            yticklabels[-2] = cbar_ends[0]
            yticklabels[1] = cbar_ends[-1]
            cbar.ax.set_yticks(yticks[1:-1], yticklabels[1:-1])
    if title is not None:
        ax.set_title(title)

    return img
