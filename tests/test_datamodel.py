import datetime

import astropy.io.fits as fits
import numpy


import numina.types.qc as qc
from numina.datamodel import DataModel


def create_test_data():
    return numpy.ones((10, 10), dtype="int32")


def create_test_image(hdr=None):
    data = create_test_data()
    hdu = fits.PrimaryHDU(data)

    hdr = {} if hdr is None else hdr
    for key, val in hdr.items():
        hdu.header[key] = val

    img = fits.HDUList([hdu])
    return img


def test_datamodel1():
    datamodel = DataModel()
    assert datamodel.name == "UNKNOWN"
    datamodel = DataModel("CLODIA")
    assert datamodel.name == "CLODIA"


def test_datamodel2():
    img = create_test_image()
    testdata = create_test_data()
    datamodel = DataModel("CLODIA")
    data = datamodel.get_data(img)

    assert numpy.allclose(data, testdata)


def test_qc():
    img = create_test_image()

    datamodel = DataModel("CLODIA")

    qcontrol = datamodel.get_quality_control(img)

    assert qcontrol == qc.QC.UNKNOWN


def test_imgid():

    CHECKSUM = "RfAdUd2cRd9cRd9c"

    hdr = {"CHECKSUM": CHECKSUM}
    img = create_test_image(hdr)

    datamodel = DataModel("CLODIA")

    imgid_chsum = datamodel.get_imgid(img)

    assert imgid_chsum == CHECKSUM


def test_ginfo():

    CHECKSUM = "RfAdUd2cRd9cRd9c"

    uuid_str = "b2f3d815-6f59-48e3-bea1-4d1ea1a3abc1"

    hdr = {
        "CHECKSUM": CHECKSUM,
        "instrume": "CLODIA",
        "object": "",
        "obsmode": "TEST",
        "numtype": "test_img",
        "exptime": 560,
        "darktime": 573,
        "uuid": uuid_str,
        "DATE-OBS": "1975-03-31T12:23:45.00",
        "blckuuid": 1,
        "insconf": "v1",
    }

    date_obs = datetime.datetime(1975, 3, 31, 12, 23, 45)

    ref = {
        "instrument": "CLODIA",
        "object": "",
        "n_ext": 1,
        "name_ext": ["PRIMARY"],
        "quality_control": qc.QC.UNKNOWN,
        "mode": "TEST",
        "type": "test_img",
        "exptime": 560,
        "darktime": 573,
        "uuid": uuid_str,
        "observation_date": date_obs,
        "blckuuid": 1,
        "block_uuid": 1,
        "imgid": uuid_str,
        "insconf": "v1",
        "insconf_uuid": "v1",
    }

    img = create_test_image(hdr)

    datamodel = DataModel("CLODIA")

    imgid_chsum = datamodel.gather_info_hdu(img)
    assert imgid_chsum == ref


def test_aliases_follow_targets():
    """The aliases have the value of their targets, also when these are redefined"""
    hdr = {"BLCKUUID": "b1", "INSCONF": "c1", "OBUUID": "b2", "CONFUUID": "c2"}
    img = create_test_image(hdr)

    datamodel = DataModel("CLODIA")
    assert datamodel.extractor.extract("block_uuid", img) == "b1"
    assert datamodel.extractor.extract("insconf_uuid", img) == "c1"

    datamodel = DataModel("CLODIA", mappings={"blckuuid": "OBUUID", "insconf": "CONFUUID"})
    for key, value in [("blckuuid", "b2"), ("block_uuid", "b2"), ("insconf", "c2"), ("insconf_uuid", "c2")]:
        assert datamodel.extractor.extract(key, img) == value


def test_gather_info_oresult():
    from numina.core import DataFrame
    from numina.core.oresult import ObservationResult

    ob = ObservationResult()
    for uuid_str in ["u1", "u2"]:
        ob.frames.append(DataFrame(frame=create_test_image({"UUID": uuid_str, "BLCKUUID": "b1"})))

    datamodel = DataModel("CLODIA")
    info = datamodel.gather_info_oresult(ob)
    assert [item["uuid"] for item in info] == ["u1", "u2"]
    assert [item["block_uuid"] for item in info] == ["b1", "b1"]
    assert datamodel.gather_info(ob.frames[0]) == info[0]
    assert list(datamodel.extractor_map) == ["fits"]
