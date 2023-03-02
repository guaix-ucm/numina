import pytest

from numina.exceptions import ValidationError
from numina.types.datatype import ListOfType, PlainPythonType


@pytest.mark.parametrize("nmin, nmax, allowed, not_allowed", [
    (None, None, [[], [1,2,3], [-1, 2.0, 34.0, 4, -5]], []),
    (1, None, [[-1, 2.0, 34.0, 4, -5]], [[]]),
    (None, 3, [[], [1], [2, 3], [1,2,3]], [[-1, 2.0, 34.0, 4, -5]]),
    (4, 4, [[-1, 2.0, 34.0, 4]], [[1, 2, 3], []]),
])

def test_list(nmin, nmax, allowed, not_allowed):
    ppt = ListOfType(PlainPythonType(ref=0), nmin=nmin, nmax=nmax)

    for obj1 in allowed:
        assert ppt.convert(obj1) == obj1

    for obj2 in not_allowed:
        with pytest.raises(ValidationError):
            ppt.convert(obj2)
