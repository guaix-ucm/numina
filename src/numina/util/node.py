#
# Copyright 2010-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


import abc
import logging

_logger = logging.getLogger(__name__)


class Node(metaclass=abc.ABCMeta):
    """An elemental operation in a Flow.

    Parameters
    ----------
    ninputs : int
        Number of inputs.
    noutputs : int
        Number of outputs.
    """

    def __init__(self, ninputs=1, noutputs=1):
        super().__init__()
        self._nin = ninputs
        self._nout = noutputs

    @property
    def ninputs(self):
        """Number of inputs of the node."""
        return self._nin

    @property
    def noutputs(self):
        """Number of outputs of the node."""
        return self._nout

    @abc.abstractmethod
    def run(self, img):
        """Run the operation of the node on its input and return the output."""
        raise NotImplementedError

    def __call__(self, img):
        """Run the node, as ``run``."""
        self.obtain_tuple(img)
        return self.run(img)

    def obtain_tuple(self, arg):
        """Return ``arg`` if it is a tuple, a tuple with ``arg`` otherwise."""
        if isinstance(arg, tuple):
            return arg
        return (arg,)

    def execute(self, arg):
        """Run the node, as ``run``."""
        return self.run(arg)


class AdaptorNode(Node):
    """A :class:`Node` that runs a function."""

    def __init__(self, work, ninputs=1, noutputs=1):
        """work is a function object"""
        super().__init__(ninputs, noutputs)
        self.work = work

    def run(self, img):
        """Return the result of the function with the input."""
        return self.work(img)


class IdNode(Node):
    """A Node that returns its inputs."""

    def __init__(self):
        """Identity"""
        super().__init__()

    def run(self, img):
        """Return the input."""
        return img


class OutputSelector(Node):
    """A Node that returns part of the results.

    Parameters
    ----------
    ninputs : int
        Number of inputs.
    indexes : sequence of int
        Indexes of the inputs returned.
    """

    def __init__(self, ninputs, indexes):
        noutputs = len(indexes)
        super().__init__(ninputs, noutputs)
        self.indexes = indexes

    def run(self, arg):
        """Return the inputs with the selected indexes, in the order of the inputs.

        A single selected input is returned alone, not in a tuple.
        """
        res = tuple(ar for idx, ar in enumerate(arg) if idx in self.indexes)
        if len(res) == 1:
            return res[0]
        return res
