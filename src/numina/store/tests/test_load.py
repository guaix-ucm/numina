from ..load import load


def test_load_base():

    class A:
        pass

    tag = A()
    obj = 0

    assert obj == load(tag, obj)


def test_load_method():

    class A:

        def _datatype_load(self, obj):
            return obj + 1

    tag = A()
    obj = 1

    assert obj + 1 == load(tag, obj)


def test_load_method_deprecated():

    class B:

        def __numina_load__(self, obj):
            return obj + 2

    tag = B()
    obj = 1

    assert obj + 2 == load(tag, obj)


def test_load_method_register():

    class C:
        pass

    def numina_load_func(tag, obj):
        return obj + 3

    load.register(C, numina_load_func)

    tag = C()
    obj = 1

    assert obj + 3 == load(tag, obj)
