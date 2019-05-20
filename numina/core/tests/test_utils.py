
import pytest
import astropy.io.fits as fits
import numpy

from ..utils import Combine
from numina.core.oresult import ObservationResult
from numina.tests.simpleobj import create_simple_frame


@pytest.mark.parametrize(
    "method,result", [
        ('mean', 5200.0), ('median', 2800.0), ('minmax', 2666.6667),
        ('sigmaclip', 5200.0)
    ])
def test_combine1(method, result):

    obresult = ObservationResult()
    values = [2000.0, 3000.0, 2200.0, 2800.0, 16000.0]
    nimages = len(values)
    obresult.frames = [
        create_simple_frame(value=value) for value in values
    ]

    recipe = Combine()

    r_input = recipe.create_input(
        obresult=obresult,
        method=method
    )

    r_result = recipe(r_input)
    assert isinstance(r_result, Combine.RecipeResult)

    hdul = r_result.result.open()

    assert isinstance(hdul, fits.HDUList)
    assert numpy.allclose(hdul[0].data, [[result]])
    assert hdul[0].header['NUM-NCOM'] == nimages


@pytest.mark.parametrize(
    "method,kwargs,result", [
        ('minmax', {'nmin': 1, 'nmax': 1}, 2624.8),
        ('minmax', {'nmin': 2, 'nmax': 2}, 2667.0),
        ('minmax', {'nmin': 3, 'nmax': 3}, 2801),
        ('sigmaclip', {'high': 1.0, 'low': 2.0}, 2170.6667)
    ])
def test_combine2(method, kwargs, result):

    obresult = ObservationResult()
    values = [-100, 3123, 2000.0, 3000.0, 2200.0, 2801.0, 16000.0]
    nimages = len(values)
    obresult.frames = [
        create_simple_frame(value=value) for value in values
    ]

    recipe = Combine()

    r_input = recipe.create_input(
        obresult=obresult,
        method=method,
        method_kwargs=kwargs
    )

    r_result = recipe(r_input)
    assert isinstance(r_result, Combine.RecipeResult)

    hdul = r_result.result.open()

    assert isinstance(hdul, fits.HDUList)

    assert numpy.allclose(hdul[0].data, [[result]])

    assert hdul[0].header['NUM-NCOM'] == nimages
