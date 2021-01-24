#
# Copyright 2010-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import abc
import logging


_logger = logging.getLogger(__name__)


class Node(metaclass=abc.ABCMeta):
    """An elemental operation in a Flow."""

    def __init__(self, ninputs=1, noutputs=1):
        super(Node, self).__init__()
        self._nin = ninputs
        self._nout = noutputs

    @property
    def ninputs(self):
        return self._nin

    @property
    def noutputs(self):
        return self._nout

    @abc.abstractmethod
    def run(self, img):
        raise NotImplementedError

    def __call__(self, img):
        _args = self.obtain_tuple(img)
        return self.run(img)

    def obtain_tuple(self, arg):
        if isinstance(arg, tuple):
            return arg
        return (arg,)

    def execute(self, arg):
        return self.run(arg)


class AdaptorNode(Node):
    """A :class:`Node` that runs a function."""
    def __init__(self, work, ninputs=1, noutputs=1):
        '''work is a function object'''
        super(AdaptorNode, self).__init__(ninputs, noutputs)
        self.work = work

    def run(self, img):
        return self.work(img)


class IdNode(Node):
    """A Node that returns its inputs."""
    def __init__(self):
        """Identity"""
        super(IdNode, self).__init__()

    def run(self, img):
        return img


class OutputSelector(Node):
    """A Node that returns part of the results."""
    def __init__(self, ninputs, indexes):
        noutputs = len(indexes)
        super(OutputSelector, self).__init__(ninputs, noutputs)
        self.indexes = indexes

    def run(self, arg):
        res = tuple(ar for idx, ar in enumerate(arg) if idx in self.indexes)
        if len(res) == 1:
            return res[0]
        return res
