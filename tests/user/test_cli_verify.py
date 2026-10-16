"""numina verify"""

import astropy.io.fits as fits
import pytest
import yaml

import numina.core.pipelineload as pload
from numina.exceptions import ValidationError
from numina.user.cli import main
from tests.support.drpdef import make_drp

DRP_TEST1 = make_drp(
    {"bias": "numina.core.utils.AlwaysSuccessRecipe", "image": "numina.core.utils.AlwaysSuccessRecipe"},
    mode_options={
        "bias": {"rawimage": "BIAS", "validator": "tests.user.test_cli_verify.validate_same_exptime"},
        "image": {"rawimage": "IMAGE"},
    },
)


def validate_same_exptime(mode, obsres):
    """Validator of the bias mode: all the frames have the same EXPTIME"""
    values = set()
    for frame in obsres.frames:
        with frame.open() as hdulist:
            values.add(hdulist[0].header["EXPTIME"])
    if len(values) > 1:
        raise ValidationError("EXPTIME is not the same in all the images")


def check_test1(obj, astype=None, level=None):
    """The checks of TEST1: OBSMODE is required, and is astype if given"""
    if isinstance(obj, dict):
        if "uuid" not in obj:
            raise ValidationError("'uuid' is missing")
        return True
    hdr = obj[0].header
    if "OBSMODE" not in hdr:
        raise ValidationError("HDU 0: 'OBSMODE' is a required property")
    if astype is not None and hdr["OBSMODE"] != astype:
        raise ValidationError(f"OBSMODE is {hdr['OBSMODE']}, not {astype}")
    return True


@pytest.fixture
def datadir(drpmocker, monkeypatch, tmp_path):
    def load_drp():
        drp = pload.drp_load_data("numina", DRP_TEST1)
        drp.checker = check_test1
        return drp

    drpmocker.add_drp("TEST1", load_drp)
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    images = {
        "bias1.fits": {"INSTRUME": "TEST1", "OBSMODE": "BIAS", "EXPTIME": 0.0},
        "bias2.fits": {"INSTRUME": "TEST1", "OBSMODE": "BIAS", "EXPTIME": 0.0},
        "bias3.fits": {"INSTRUME": "TEST1", "OBSMODE": "BIAS", "EXPTIME": 1.0},
        "image1.fits": {"INSTRUME": "TEST1", "OBSMODE": "IMAGE", "EXPTIME": 10.0},
        "broken.fits": {"INSTRUME": "TEST1"},
        "other.fits": {"INSTRUME": "OTHER"},
        "noinstrument.fits": {},
    }
    for name, cards in images.items():
        hdr = fits.Header()
        hdr.update(cards)
        fits.PrimaryHDU(header=hdr).writeto(data / name)
    return data


def run_verify(capsys, *args):
    status = main(["--disable-plugins", "verify", *args])
    lines = capsys.readouterr().out.splitlines()
    return status, lines


def test_verify_files(datadir, capsys):
    status, lines = run_verify(
        capsys, "data/bias1.fits", "data/broken.fits", "data/other.fits", "data/noinstrument.fits", "notes.txt"
    )
    assert status == 1
    assert lines == [
        "OK data/bias1.fits",
        "INVALID data/broken.fits: HDU 0: 'OBSMODE' is a required property",
        "NOT CHECKED data/other.fits: no checks for instrument OTHER",
        "NOT CHECKED data/noinstrument.fits: no INSTRUME in the primary header",
        "NOT CHECKED notes.txt: not a FITS or JSON file",
        "1 valid, 1 invalid, 3 not checked",
    ]


def test_verify_valid_files(datadir, capsys):
    status, lines = run_verify(capsys, "data/bias1.fits", "data/image1.fits")
    assert status == 0
    assert lines[-1] == "2 valid, 0 invalid, 0 not checked"


def test_verify_missing_file(datadir, capsys):
    status, lines = run_verify(capsys, "data/missing.fits")
    assert status == 1
    assert lines[0].startswith("INVALID data/missing.fits: FileNotFoundError")


def test_verify_json(datadir, capsys, tmp_path):
    (tmp_path / "good.json").write_text('{"instrument": "TEST1", "uuid": "1"}')
    (tmp_path / "bad.json").write_text('{"instrument": "TEST1"}')
    (tmp_path / "noinstrument.json").write_text("{}")
    status, lines = run_verify(capsys, "good.json", "bad.json", "noinstrument.json")
    assert status == 1
    assert lines[:3] == [
        "OK good.json",
        "INVALID bad.json: 'uuid' is missing",
        "NOT CHECKED noinstrument.json: no instrument field",
    ]


@pytest.mark.parametrize("mode", ["bias", "TEST1.bias"])
def test_verify_as_mode(datadir, capsys, mode):
    status, lines = run_verify(capsys, "--mode", mode, "data/bias1.fits", "data/image1.fits", "data/other.fits")
    assert status == 1
    assert lines[:3] == [
        "OK data/bias1.fits",
        "INVALID data/image1.fits: OBSMODE is IMAGE, not BIAS",
        "INVALID data/other.fits: the instrument is OTHER, the mode bias is of TEST1",
    ]


def test_verify_unknown_mode(datadir, capsys):
    assert main(["--disable-plugins", "verify", "--mode", "nomode", "data/bias1.fits"]) == 2
    assert "no mode nomode in any DRP" in capsys.readouterr().err


def test_verify_ob_and_mode_not_allowed(datadir):
    with pytest.raises(SystemExit) as excinfo:
        main(["--disable-plugins", "verify", "--ob", "--mode", "bias", "ob.yaml"])
    assert excinfo.value.code == 2


def write_obs(tmp_path, *docs):
    path = tmp_path / "obs.yaml"
    path.write_text(yaml.safe_dump_all(docs))
    return path.name


def test_verify_ob(datadir, capsys, tmp_path):
    obs = write_obs(
        tmp_path,
        {"id": 1, "instrument": "TEST1", "mode": "bias", "frames": ["bias1.fits", "bias2.fits"]},
        {"id": 2, "instrument": "TEST1", "mode": "image", "images": ["image1.fits"]},
    )
    status, lines = run_verify(capsys, "--ob", "--datadir", "data", obs)
    assert status == 0
    assert lines == [
        "OK data/bias1.fits",
        "OK data/bias2.fits",
        "OK obs.yaml (id 1)",
        "OK data/image1.fits",
        "NOT CHECKED obs.yaml (id 2): no validator for mode image",
        "4 valid, 0 invalid, 1 not checked",
    ]


def test_verify_ob_invalid(datadir, capsys, tmp_path):
    obs = write_obs(
        tmp_path,
        {"id": 1, "instrument": "TEST1", "mode": "bias", "frames": ["bias1.fits", "image1.fits"]},
        {"id": 2, "instrument": "TEST1", "mode": "bias", "frames": ["bias1.fits", "bias3.fits"]},
        {"id": 3, "instrument": "TEST1", "mode": "nomode", "frames": ["bias1.fits"]},
    )
    status, lines = run_verify(capsys, "--ob", "--datadir", "data", obs)
    assert status == 1
    assert lines == [
        "OK data/bias1.fits",
        "INVALID data/image1.fits: OBSMODE is IMAGE, not BIAS",
        "NOT CHECKED obs.yaml (id 1): some raw images are not valid",
        "OK data/bias1.fits",
        "OK data/bias3.fits",
        "INVALID obs.yaml (id 2): EXPTIME is not the same in all the images",
        "INVALID obs.yaml (id 3): no mode nomode in the DRP of TEST1",
        "3 valid, 3 invalid, 1 not checked",
    ]


def test_verify_ob_datadir_from_config(datadir, capsys, tmp_path):
    obs = write_obs(tmp_path, {"id": 1, "instrument": "TEST1", "mode": "image", "frames": ["image1.fits"]})
    # the default datadir in [tool.run] is data
    status, lines = run_verify(capsys, "--ob", obs)
    assert lines[0] == "OK data/image1.fits"
