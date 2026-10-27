"""Flows of nodes of numina.util.flow"""

import pytest

from numina.util.flow import FlowError, MixerFlow, ParallelFlow, SerialFlow
from numina.util.node import AdaptorNode, IdNode


def add(value):
    return AdaptorNode(lambda x: x + value)


def test_serial_flow():
    flow = SerialFlow([add(1), add(10), IdNode()])

    assert flow(1) == 12
    assert len(flow) == 3
    assert isinstance(flow[2], IdNode)
    assert [node(0) for node in flow] == [1, 10, 0]

    flow[2] = add(100)
    assert flow(1) == 112


def test_serial_flow_empty():
    flow = SerialFlow([])
    assert (flow.ninputs, flow.noutputs) == (1, 1)
    assert flow(5) == 5


def test_serial_flow_inputs_outputs():
    split = AdaptorNode(lambda x: (x, -x), noutputs=2)
    join = AdaptorNode(lambda args: args[0] * args[1], ninputs=2)
    flow = SerialFlow([split, join])
    assert (flow.ninputs, flow.noutputs) == (1, 1)
    assert flow(3) == -9


def test_serial_flow_error():
    split = AdaptorNode(lambda x: (x, -x), noutputs=2)
    with pytest.raises(FlowError):
        SerialFlow([split, add(1)])


def test_parallel_flow():
    flow = ParallelFlow([add(1), add(10)])

    assert (flow.ninputs, flow.noutputs) == (2, 2)
    assert flow((1, 2)) == (2, 12)
    assert len(flow) == 2
    assert [node(0) for node in flow] == [1, 10]

    flow[0] = IdNode()
    assert isinstance(flow[0], IdNode)
    assert flow((1, 2)) == (1, 12)


def test_parallel_flow_several_outputs():
    split = AdaptorNode(lambda x: (x, -x), noutputs=2)
    flow = ParallelFlow([split, AdaptorNode(lambda x: (x * 2,))])

    assert (flow.ninputs, flow.noutputs) == (2, 3)
    assert flow((1, 2)) == (1, -1, 4)


def test_mixer_flow():
    flow = MixerFlow([2, 0, 0])
    assert (flow.ninputs, flow.noutputs) == (3, 3)
    assert flow(("a", "b", "c")) == ("c", "a", "a")

    with pytest.raises(AssertionError):
        flow(("a", "b"))
