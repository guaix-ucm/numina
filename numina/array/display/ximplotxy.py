from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np

from .pause_debugplot import pause_debugplot


def ximplotxy(x, y, plottype=None,
              xlim=None, ylim=None, 
              xlabel=None, ylabel=None, title=None,
              show=True, debugplot=0,
              **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if plottype == 'semilog':
        ax.semilogy(x, y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(debugplot)
    else:
        # return axes
        return ax


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='ximplotxy')
    parser.add_argument("filename",
                        help="ASCII file with data in columns")
    parser.add_argument("cols",
                        help="Tuple col1 col2",
                        type=int, nargs=2)
    parser.add_argument("--kwargs",
                        help="Extra arguments for plot, e.g.: "
                             "{'marker':'o',"
                             " 'linestyle':'dotted',"
                             " 'xlabel':'x axis', 'ylabel':'y axis',"
                             " 'title':'sample plot',"
                             " 'xlim':[-1,1], ylim:[-2,2],"
                             " 'label':'sample data',"
                             " 'color':'magenta'}")
    args = parser.parse_args(args)

    # ASCII file
    filename = args.filename

    # columns to be plotted (first column will be number 1 and not 0)
    col1, col2 = args.cols
    col1 -= 1
    col2 -= 1

    # read ASCII file
    bigtable = np.genfromtxt(filename)
    x = bigtable[:, col1]
    y = bigtable[:, col2]

    if args.kwargs is None:
        ximplotxy(x, y, debugplot=12, marker='o', linestyle='')
    else:
        ximplotxy(x, y, debugplot=12, **eval(args.kwargs))


if __name__ == '__main__':
    main()
