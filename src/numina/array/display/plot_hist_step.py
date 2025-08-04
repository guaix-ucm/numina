#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Plot histogram using the matplotlib step function."""
import numpy as np


def plot_hist_step(ax, bins, h, label=None,
                   color='C0', alpha=1.0, fill_color=None, fill_alpha=0.4):
    """Plot a histogram using the step function.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    bins : array_like
        The bin edges for the histogram.
    h : array_like
        The histogram values.
    label : str, optional
        Label for the histogram. Default is None.
    color : str, optional
        Color of the step line. Default is 'C0'.
    alpha : float, optional
        Transparency of the step line. Default is 1.0.
    fill_color : str, optional
        Color to fill the area under the step line. Default is None (no fill).
    fill_alpha : float, optional
        Transparency of the filled area. Default is 0.4.
    """
    xdum = (bins[:-1] + bins[1:]) / 2
    ax.step(xdum, h, where='mid')
    ax.plot([bins[0], bins[0], xdum[0]], [0, h[0], h[0]], alpha=alpha, color=f'{color}', linestyle='-')
    ax.plot([xdum[-1], bins[-1], bins[-1]], [h[-1], h[-1], 0], alpha=alpha, color=f'{color}', linestyle='-')
    if fill_color is not None:
        ax.fill_between(np.concatenate((np.array([bins[0]]), xdum, np.array([bins[-1]]))),
                        np.concatenate((np.array([h[0]]), h, np.array([h[-1]]))),
                        step='mid', alpha=fill_alpha, color=f'{fill_color}')
