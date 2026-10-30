#
# Copyright 2010-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import logging

import numina.util.node as node

_logger = logging.getLogger(__name__)


class FlowError(Exception):
    """Error base class for flows."""

    pass


class SerialFlow(node.Node):
    """A flow where Nodes are executed sequentially.

    Parameters
    ----------
    nodeseq : list of Node
        The nodes, the number of outputs of each node must be the number
        of inputs of the next one.

    Raises
    ------
    FlowError
        If the outputs of a node do not match the inputs of the next one.
    """

    def __init__(self, nodeseq):
        # Checking inputs and out puts are correct
        self.nodeseq = nodeseq
        if nodeseq:
            ninputs = nodeseq[0].ninputs
            noutputs = nodeseq[-1].noutputs
            # Check nodes
            for i, o in zip(nodeseq, nodeseq[1:]):
                if i.noutputs != o.ninputs:
                    raise FlowError
        else:
            ninputs = 1
            noutputs = 1
        super().__init__(ninputs, noutputs)

    def __iter__(self):
        return self.nodeseq.__iter__()

    def __len__(self):
        return self.nodeseq.__len__()

    def __getitem__(self, key):
        return self.nodeseq[key]

    def __setitem__(self, key, value):
        self.nodeseq[key] = value

    def run(self, img):
        """Pass ``img`` through the nodes in order and return the output of the last one."""
        out = img
        for nd in self.nodeseq:
            out = nd(out)
        return out


class ParallelFlow(node.Node):
    """A flow where Nodes are executed in parallel.

    Parameters
    ----------
    nodeseq : list of Node
        The nodes. The inputs and outputs of the flow are those of the
        nodes, concatenated.
    """

    def __init__(self, nodeseq):
        self.nodeseq = nodeseq
        nin = sum((f.ninputs for f in nodeseq), 0)
        nout = sum((f.noutputs for f in nodeseq), 0)
        super().__init__(nin, nout)

    def run(self, args):
        """Run each node with its item of ``args`` and return the outputs in a tuple.

        If some node has more than one output, the outputs of all the nodes
        are concatenated in the tuple.
        """
        out = []
        for func, arg in zip(self.nodeseq, args):
            r = func(arg)
            out.append(r)

        if any(f.noutputs != 1 for f in self.nodeseq):
            return tuple(item for sublist in out for item in sublist)

        return tuple(out)

    def __iter__(self):
        return self.nodeseq.__iter__()

    def __len__(self):
        return self.nodeseq.__len__()

    def __getitem__(self, key):
        return self.nodeseq[key]

    def __setitem__(self, key, value):
        self.nodeseq[key] = value


class MixerFlow(node.Node):
    """Node that selects, reorders and repeats its inputs.

    Parameters
    ----------
    table : list of int
        Index of the input returned in each output.
    """

    def __init__(self, table):
        nin = max(table) + 1
        nout = len(table)
        super().__init__(nin, nout)
        self.table = table

    def run(self, args):
        """Return the inputs in the order of the table."""
        assert len(args) == self.ninputs

        return tuple(args[idx] for idx in self.table)
