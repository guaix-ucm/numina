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
                   linestyle='-',
                   color='C0', alpha=1.0,
                   fill_color=None, fill_alpha=0.4,
                   left_border=True, right_border=True):
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
    linestyle : str, optional
        Line style for the step line. Default is '-'.
    color : str, optional
        Color of the step line. Default is 'C0'.
    alpha : float, optional
        Transparency of the step line. Default is 1.0.
    fill_color : str, optional
        Color to fill the area under the step line. Default is None (no fill).
    fill_alpha : float, optional
        Transparency of the filled area. Default is 0.4.
    left_border : bool, optional
        If True, draw a vertical line at the left edge of the first bin.
        Default is True.
    right_border : bool, optional
        If True, draw a vertical line at the right edge of the last bin.
        Default is True.
    """
    xdum = (bins[:-1] + bins[1:]) / 2
    ax.step(xdum, h, where='mid', color=color, linestyle=linestyle, label=label)
    if left_border:
        ax.plot([bins[0], bins[0], xdum[0]], [0, h[0], h[0]], alpha=alpha, color=f'{color}', linestyle=linestyle)
    if right_border:
        ax.plot([xdum[-1], bins[-1], bins[-1]], [h[-1], h[-1], 0], alpha=alpha, color=f'{color}', linestyle=linestyle)
    if fill_color is not None:
        xfill = xdum
        hfill = h
        if left_border:
            xfill = np.concatenate((np.array([bins[0]]), xdum))
            hfill = np.concatenate((np.array([h[0]]), h))
        if right_border:
            xfill = np.concatenate((xdum, np.array([bins[-1]])))
            hfill = np.concatenate((h, np.array([h[-1]])))
        ax.fill_between(xfill, hfill, step='mid', alpha=fill_alpha, color=f'{fill_color}')
