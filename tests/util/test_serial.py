import numina.util.flow as flow


def test_serial_empty_is_id():

    empty_serial = flow.SerialFlow([])

    class Something:
        pass

    inp = Something()

    assert inp is empty_serial.run(inp)
