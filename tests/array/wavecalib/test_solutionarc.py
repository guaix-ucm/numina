import pytest

from numina.array.wavecalib.arccalibration import WavecalFeature

features = [
    {
        "category": "A",
        "xpos": 100.0,
        "reference": 3210.0,
        "wavelength": 3100.0,
        "line_ok": True,
        "ypos": 0.0,
        "peak": 0.0,
        "funcost": 12.0,
        "fwhm": 0.0,
        "lineid": 1,
    },
    {
        "category": "A",
        "xpos": 150.0,
        "reference": 3310.0,
        "wavelength": 3150.0,
        "line_ok": True,
        "ypos": 0.0,
        "peak": 0.0,
        "funcost": 12.0,
        "fwhm": 0.0,
        "lineid": 2,
    },
    {
        "category": "C",
        "xpos": 250.0,
        "reference": 3410.0,
        "wavelength": 3250.0,
        "line_ok": True,
        "ypos": 0.0,
        "peak": 0.0,
        "funcost": 13.0,
        "fwhm": 0.0,
        "lineid": 3,
    },
    {
        "category": "X",
        "xpos": 250.0,
        "reference": 3410.0,
        "wavelength": 3250.0,
        "line_ok": True,
        "ypos": 0.0,
        "peak": 0.0,
        "funcost": float("inf"),
        "fwhm": 0.0,
        "lineid": 4,
    },
]


def create_features(orig):

    # Create Features
    features = []
    for feature in orig["features"]:
        m = WavecalFeature(**feature)
        features.append(m)
    return features


def test_serialize1():

    feature0 = features[0]

    mm = WavecalFeature(**feature0)
    assert feature0 == mm.__getstate__()


def test_serialize2():
    feature3 = features[3]
    mm = WavecalFeature(**feature3)
    feature3_c = feature3.copy()
    feature3_c["funcost"] = 1e50
    with pytest.warns(RuntimeWarning, match="Converting funcost=inf"):
        assert feature3_c == mm.__getstate__()
