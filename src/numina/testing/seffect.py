"""Side effect decorator

Decorates a function to call another
"""

import warnings

warnings.warn(
    "numina.testing.seffect is deprecated and will be removed in the next version of numina, "
    "it is only used by the tests of numina",
    DeprecationWarning,
    stacklevel=2,
)


class FuncCall:
    """Record a function call"""

    def __init__(self):
        self.clear()

    def __call__(self, *args, **kwargs):
        """Record that the object has been called and the arguments of the call."""
        self.called = True
        self.args = args
        self.kwargs = kwargs

    def clear(self):
        """Forget the recorded call."""
        self.called = False
        self.args = ()
        self.kwargs = {}


def side_effect(effect_func):
    """Return a decorator that calls ``effect_func`` before the decorated function.

    The exceptions raised by ``effect_func`` are ignored.
    """
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
    """Decorate ``func`` so that its calls are recorded in a ``FuncCall``, in the attribute ``side_effect``."""

    rcd = side_effect(FuncCall())
    return rcd(func)
