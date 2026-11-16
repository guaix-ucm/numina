"""Tests for the command-line script numina.array.wavecalib.check_wlcalib.main"""

from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import numpy as np
import pytest

import numina.array.wavecalib.check_wlcalib as check_wlcalib

WV_MASTER = np.linspace(4050, 4950, 15)
NAXIS1 = 1000


def arc_spectrum():
    """Arc spectrum with lines in WV_MASTER, CRPIX1=1, CRVAL1=4000, CDELT1=1"""
    wv = 4000.0 + np.arange(NAXIS1)
    sp = np.full_like(wv, 10.0)
    for wv_line in WV_MASTER:
        sp += 1000 * np.exp(-0.5 * ((wv - wv_line) / 1.5) ** 2)
    return sp


def write_fits(filename, data, cunit1=None, wcs=True):
    header = fits.Header()
    if wcs:
        header["CRPIX1"] = 1.0
        header["CRVAL1"] = 4000.0
        header["CDELT1"] = 1.0
    if cunit1 is not None:
        header["CUNIT1"] = cunit1
    fits.writeto(filename, data, header=header)
    return filename


def write_ecsv(filename, wavelengths, unit=None):
    nlines = len(wavelengths)
    table = Table([wavelengths, np.ones(nlines, dtype=int), np.ones(nlines)], names=["wavelength", "flag", "intensity"])
    if unit is not None:
        table["wavelength"].unit = unit
    table.write(filename, format="ascii.ecsv")
    return filename


@pytest.fixture
def calls(monkeypatch):
    """Replace check_wlcalib_sp by a function storing its arguments"""
    calls = []
    monkeypatch.setattr(check_wlcalib, "check_wlcalib_sp", lambda **kwargs: calls.append(kwargs))
    return calls


@pytest.fixture
def ecsv_file(tmp_path):
    return write_ecsv(tmp_path / "lines.ecsv", WV_MASTER, unit=u.AA)


def test_1d_image_ecsv(tmp_path, calls, ecsv_file):
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum())
    check_wlcalib.main([str(fitsfile), "--scans", "1", "--wv-master-file", str(ecsv_file)])
    assert len(calls) == 1
    kwargs = calls[0]
    assert np.allclose(kwargs["sp"], arc_spectrum())
    assert (kwargs["crpix1"], kwargs["crval1"], kwargs["cdelt1"]) == (1.0, 4000.0, 1.0)
    assert np.allclose(kwargs["wv_master"], WV_MASTER)
    assert kwargs["wv_unit"] == u.AA
    assert kwargs["pdf"] is None
    assert kwargs["debugplot"] == 12


@pytest.mark.parametrize("method, func", [([], np.median), (["--method", "mean"], np.mean)])
def test_2d_image(tmp_path, calls, ecsv_file, method, func):
    rng = np.random.default_rng(1234)
    image2d = np.tile(arc_spectrum(), (4, 1)) + rng.normal(0, 5, (4, NAXIS1))
    fitsfile = write_fits(tmp_path / "sp2d.fits", image2d)
    check_wlcalib.main([str(fitsfile), "--scans", "2,4", "--wv-master-file", str(ecsv_file)] + method)
    assert np.allclose(calls[0]["sp"], func(image2d[1:4], axis=0))


def test_txt_line_list(tmp_path, calls):
    # one-column text file: Angstrom is assumed
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum())
    txtfile = tmp_path / "lines.txt"
    np.savetxt(txtfile, WV_MASTER)
    check_wlcalib.main([str(fitsfile), "--scans", "1", "--wv-master-file", str(txtfile)])
    assert np.allclose(calls[0]["wv_master"], WV_MASTER)
    assert calls[0]["wv_unit"] == u.AA


def test_unit_conversion(tmp_path, calls, ecsv_file):
    # line list in Angstrom, spectrum calibrated in nm
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum(), cunit1="nm")
    check_wlcalib.main([str(fitsfile), "--scans", "1", "--wv-master-file", str(ecsv_file)])
    assert np.allclose(calls[0]["wv_master"], WV_MASTER / 10)
    assert calls[0]["wv_unit"] == u.nm


def test_same_unit_no_conversion(tmp_path, calls, ecsv_file):
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum(), cunit1="Angstrom")
    check_wlcalib.main([str(fitsfile), "--scans", "1", "--wv-master-file", str(ecsv_file)])
    assert np.allclose(calls[0]["wv_master"], WV_MASTER)
    assert calls[0]["wv_unit"] == u.AA


def test_incompatible_units(tmp_path, calls, ecsv_file):
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum(), cunit1="s")
    with pytest.raises(ValueError, match="cannot be converted to CUNIT1"):
        check_wlcalib.main([str(fitsfile), "--scans", "1", "--wv-master-file", str(ecsv_file)])
    assert calls == []


def test_missing_wcs(tmp_path, calls, ecsv_file):
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum(), wcs=False)
    with pytest.raises(ValueError, match="Keyword CRPIX1 not found"):
        check_wlcalib.main([str(fitsfile), "--scans", "1", "--wv-master-file", str(ecsv_file)])


def test_old_option_names(tmp_path, calls, ecsv_file):
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum())
    check_wlcalib.main(
        [
            str(fitsfile),
            "--scans",
            "1",
            "--wv_master_file",
            str(ecsv_file),
            "--nwinwidth_initial",
            "9",
            "--ntimes_match_wv=3",
            "--poldeg_residuals",
            "2",
            "--remove_null_borders",
        ]
    )
    kwargs = calls[0]
    assert kwargs["nwinwidth_initial"] == 9
    assert kwargs["ntimes_match_wv"] == 3.0
    assert kwargs["poldeg_residuals"] == 2
    assert kwargs["remove_null_borders"] is True


def test_translate_renamed_options():
    argv = ["file.fits", "--wv_master_file", "a.ecsv", "--times_sigma_reject=4", "--use_r", "--scans", "1"]
    assert check_wlcalib._translate_renamed_options(argv) == [
        "file.fits",
        "--wv-master-file",
        "a.ecsv",
        "--times-sigma-reject=4",
        "--use-r",
        "--scans",
        "1",
    ]


def test_pdf_output(tmp_path, ecsv_file):
    # complete execution (without replacing check_wlcalib_sp)
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum())
    pdffile = tmp_path / "check.pdf"
    check_wlcalib.main(
        [
            str(fitsfile),
            "--scans",
            "1",
            "--wv-master-file",
            str(ecsv_file),
            "--pdffile",
            str(pdffile),
            "--debugplot",
            "0",
        ]
    )
    assert pdffile.read_bytes().startswith(b"%PDF")


def test_interactive_and_pdf(tmp_path, calls, ecsv_file):
    fitsfile = write_fits(tmp_path / "sp1d.fits", arc_spectrum())
    pdffile = tmp_path / "check.pdf"
    with pytest.raises(SystemExit):
        check_wlcalib.main(
            [
                str(fitsfile),
                "--scans",
                "1",
                "--wv-master-file",
                str(ecsv_file),
                "--pdffile",
                str(pdffile),
                "--interactive",
            ]
        )
    assert not pdffile.exists()
    assert calls == []
