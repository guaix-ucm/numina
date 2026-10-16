"""The instrument configuration of an observation result"""

import astropy.io.fits as fits
import pytest

from numina.core import DataFrame, ObservationResult
from numina.core.pipelineload import drp_load_data
from numina.instrument.assembly import configure_obresult
from numina.instrument.device import DeviceBase

from tests.support.drpdef import make_drp

# configurations of TEST1, the old one valid until 2016-06-01
TEST1_UUID = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
TEST1_OLD_UUID = "6ad5dc90-6b15-43b7-abb5-b07340e19f41"


@pytest.fixture(scope="module")
def drp():
    return drp_load_data("numina", make_drp())


@pytest.fixture
def configured_with(monkeypatch):
    """The images used to configure the instrument configurations"""
    images = []
    configure = DeviceBase.configure_with_image

    def record(self, image):
        if self.parent is None:
            images.append(image)
        configure(self, image)

    monkeypatch.setattr(DeviceBase, "configure_with_image", record)
    return images


def obresult(date_obs=None, instrument="TEST1"):
    ob = ObservationResult(instrument=instrument)
    if date_obs is not None:
        hdr = fits.Header({"INSTRUME": "TEST1", "DATE-OBS": date_obs})
        ob.frames = [DataFrame(frame=fits.HDUList([fits.PrimaryHDU(header=hdr)]))]
    return ob


@pytest.mark.parametrize(
    "date_obs, uuid", [("2017-01-01T00:00:00", TEST1_UUID), ("2015-01-01T00:00:00", TEST1_OLD_UUID)]
)
def test_configuration_from_frame(drp, configured_with, date_obs, uuid):
    ob = obresult(date_obs)
    configuration = configure_obresult(ob, drp)
    assert ob.configuration is configuration
    assert str(configuration.origin.uuid) == uuid
    assert ob.profile == uuid
    # configured with the frame
    assert configured_with == [ob.frames[0].frame]


def test_configuration_from_profile(drp):
    """The profile of the OB is used instead of the frames"""
    ob = obresult("2017-01-01T00:00:00")
    ob.profile = TEST1_OLD_UUID
    assert str(configure_obresult(ob, drp).origin.uuid) == TEST1_OLD_UUID


def test_configuration_without_frames(drp, configured_with):
    """Selected by the name of the instrument, not configured"""
    ob = obresult()
    configuration = configure_obresult(ob, drp)
    assert configuration.name == "TEST1"
    assert configured_with == []


def test_configuration_other_instrument(drp):
    ob = obresult("2017-01-01T00:00:00", instrument="TEST2")
    with pytest.raises(ValueError, match="is for instrument 'TEST1', not 'TEST2'"):
        configure_obresult(ob, drp)
