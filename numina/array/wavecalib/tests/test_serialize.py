
from ..arccalibration import SolutionArcCalibration, WavecalFeature, CrLinear


def create_solution(orig):

    # Create Features
    features = []
    for feature in orig['features']:
        features.append(
            WavecalFeature(**feature)
        )

    cr_linear = CrLinear(**orig['cr_linear'])
    mm = SolutionArcCalibration(features, orig['coeff'], orig['residual_std'], cr_linear)
    return mm


def test_serialize1():

    orig = {'cr_linear': {'cdelt': 3.0, 'crmax': 4600, 'crmin': 2300, 'crpix': 1200,
                          'crval': 12},
            'features': [
        {'category': 'A', 'xpos': 100, 'wv': 3210, 'line_ok': True, 'ypos': 0, 'flux': 0, 'funcost': 12.0, 'fwhm': 0,
         'lineid': 1},
        {'category': 'A', 'xpos': 150, 'wv': 3310, 'line_ok': True, 'ypos': 0, 'flux': 0, 'funcost': 12.0, 'fwhm': 0,
         'lineid': 2},
        {'category': 'C', 'xpos': 250, 'wv': 3410, 'line_ok': True, 'ypos': 0, 'flux': 0, 'funcost': 13.0, 'fwhm': 0,
         'lineid': 3}], 'wavelength': [11.0, 16.0, 26.0], 'coeff': [1.0, 0.1], 'residual_std': 1.0}

    mm = create_solution(orig)
    assert orig == mm.__getstate__()


def test_serialize2():

    orig = {'cr_linear': {'cdelt': 3.0, 'crmax': 4600, 'crmin': 2300, 'crpix': 1200,
                          'crval': 12},
            'features': [
        {'category': 'A', 'xpos': 100, 'wv': 3210, 'line_ok': True, 'ypos': 0, 'flux': 0, 'funcost': 12.0, 'fwhm': 0,
         'lineid': 1},
        {'category': 'A', 'xpos': 150, 'wv': 3310, 'line_ok': True, 'ypos': 0, 'flux': 0, 'funcost': 12.0, 'fwhm': 0,
         'lineid': 2},
        {'category': 'C', 'xpos': 250, 'wv': 3410, 'line_ok': True, 'ypos': 0, 'flux': 0, 'funcost': 13.0, 'fwhm': 0,
         'lineid': 3}], 'wavelength': [11.0, 16.0, 26.0], 'coeff': [1.0, 0.1], 'residual_std': 1.0}

    mm = create_solution(orig)

    reconstructed = SolutionArcCalibration.__new__(SolutionArcCalibration)

    reconstructed.__setstate__(orig)

    assert reconstructed == mm