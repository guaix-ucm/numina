
"""Side effect decorator

Decorates a function to call another
"""


class FuncCall(object):
    """Record a function call"""
    def __init__(self):
        self.clear()

    def __call__(self, *args, **kwargs):
        self.called = True
        self.args = args
        self.kwargs = kwargs

    def clear(self):
        self.called = False
        self.args = ()
        self.kwargs = {}


def side_effect(effect_func):
    import functools
    def side_effect_decorator(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            try:
                effect_func(*args, **kwargs)
            except Exception:
                # ignoring side effect exceptions
                pass
            return func(*args, **kwargs)

        func_wrapper.side_effect = effect_func

        return func_wrapper
    return side_effect_decorator


def record_call(func):

    rcd = side_effect(FuncCall())
    return rcd(func)
