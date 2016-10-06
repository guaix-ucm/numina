
from ..arccalibration import SolutionArcCalibration, WavecalFeature, CrLinear

orig = {'cr_linear': {'cdelt': 3.0, 'crmax': 4600.0, 'crmin': 2300.0,
                      'crpix': 1200.0,
                      'crval': 12.0},
        'features': [
            {'category': 'A', 'xpos': 100.0, 'reference': 3210.0,
             'wavelength': 3100.0, 'line_ok': True, 'ypos': 0.0, 'peak': 0.0,
             'funcost': 12.0, 'fwhm': 0.0,
             'lineid': 1},
            {'category': 'A', 'xpos': 150.0, 'reference': 3310.0,
             'wavelength': 3150.0, 'line_ok': True, 'ypos': 0.0, 'peak': 0.0,
             'funcost': 12.0, 'fwhm': 0.0,
             'lineid': 2},
            {'category': 'C', 'xpos': 250.0, 'reference': 3410.0,
             'wavelength': 3250.0, 'line_ok': True, 'ypos': 0.0, 'peak': 0.0,
             'funcost': 13.0, 'fwhm': 0.0,
             'lineid': 3}], 'coeff': [3000.0, 1.0], 'residual_std': 1.0}


def create_solution(orig):

    # Create Features
    features = []
    for feature in orig['features']:
        m = WavecalFeature(**feature)
        features.append(m)

    cr_linear = CrLinear(**orig['cr_linear'])
    mm = SolutionArcCalibration(features, orig['coeff'], orig['residual_std'], cr_linear)
    return mm


def test_serialize1():

    mm = create_solution(orig)
    assert orig == mm.__getstate__()


def test_serialize2():

    mm = create_solution(orig)

    reconstructed = SolutionArcCalibration.__new__(SolutionArcCalibration)

    reconstructed.__setstate__(orig)

    assert reconstructed == mm
