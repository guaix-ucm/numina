from numina.store.dump import dump


def test_dump_base():

    class A:
        pass

    tag = A()
    obj = 0
    where = ""

    assert obj == dump(tag, obj, where)


def test_dump_method():

    class A:

        def _datatype_dump(self, obj, where):
            return obj + 1

    tag = A()
    obj = 1
    where = ""

    assert obj + 1 == dump(tag, obj, where)


def test_dump_method_deprecated():

    class B:

        def __numina_dump__(self, obj, where):
            return obj + 2

    tag = B()
    obj = 1
    where = ""

    assert obj + 2 == dump(tag, obj, where)


def test_dump_method_register():

    class C:
        pass

    def numina_dump_func(tag, obj, where):
        return obj + 3

    dump.register(C, numina_dump_func)

    tag = C()
    obj = 1
    where = ""

    assert obj + 3 == dump(tag, obj, where)
