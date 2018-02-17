

from .. import SerialFlow


def test_serial_empty_is_id():

    empty_serial = SerialFlow([])

    class Something(object):
        pass

    inp = Something()

    assert inp is empty_serial.run(inp)
