#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import datetime
import logging

from astropy.io import fits

import numina.array as array
import numina.util.node as node
import numina.datamodel as dm

_logger = logging.getLogger(__name__)


def promote_hdulist(hdulist, totype="float32"):
    """Return a new HDUList with the data of the image HDUs converted to ``totype``.

    Each HDU is converted with ``promote_hdu``.
    """
    nn = [promote_hdu(hdu, totype=totype) for hdu in hdulist]
    newhdulist = fits.HDUList(nn)
    return newhdulist


def promote_hdu(hdu, totype="float32"):
    """Return a copy of a primary or image HDU with its data converted to ``totype``.

    The header is copied. HDUs without data and HDUs of other types are
    returned unchanged.
    """
    if hdu.data is None:
        return hdu
    newdata = hdu.data.astype(totype)
    newheader = hdu.header.copy()
    if isinstance(hdu, fits.PrimaryHDU):
        hdu = fits.PrimaryHDU(newdata, header=newheader)
        return hdu
    elif isinstance(hdu, fits.ImageHDU):
        hdu = fits.ImageHDU(newdata, header=newheader)
        return hdu
    else:
        # do nothing
        pass
    return hdu


class SimpleDataModel(dm.DataModel):
    """Model of the Data being processed"""

    pass


class Corrector(node.Node):
    """A Node that corrects a frame from instrumental signatures.

    Parameters
    ----------
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data and headers of the frame.
        If not given, a ``SimpleDataModel`` is used.
    calibid : str
        Identifier of the calibration, recorded in the header of the corrected frame.
    dtype : str
        Data type of the corrected data. Integer images are promoted to it before
        ``run``, and the subclasses that convert the data use it.
    """

    def __init__(self, datamodel=None, calibid="calibid-unknown", dtype="float32"):
        super().__init__()
        if not datamodel:
            self.datamodel = SimpleDataModel()
        else:
            self.datamodel = datamodel
        self.dtype = dtype
        self.calibid = calibid

    def __call__(self, img):
        """Correct ``img``, promoting integer data to ``dtype`` before calling ``run``."""
        if img[0].data.dtype.kind in "iu":
            # The corrections are done in place, integer data
            # (such as the uint16 images of GCS) are promoted to self.dtype
            _logger.debug("change dtype to %s, old is %s", self.dtype, img[0].data.dtype)
            img = promote_hdulist(img, totype=self.dtype)
        img = self.run(img)

        return img

    def get_imgid(self, img, prefix=False):
        """Return the identifier of the image, computed with ``numina.datamodel.get_imgid``."""
        return dm.get_imgid(img, prefix=prefix)


TagOptionalCorrector = Corrector


class BadPixelCorrector(Corrector):
    """A Node that corrects a frame from bad pixels.

    Parameters
    ----------
    badpixelmask : numpy.ndarray
        Bad pixel mask, nonzero values mark the bad pixels.
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data of the frame.
    calibid : str
        Identifier of the mask, recorded in the header of the corrected frame.
    dtype : str
        Data type to which integer images are promoted, float images keep their type.
    hwin, wwin : int
        Half height and half width of the window used to compute the median
        that replaces each bad pixel.
    """

    def __init__(self, badpixelmask, datamodel=None, calibid="calibid-unknown", dtype="float32", hwin=2, wwin=2):

        super().__init__(datamodel, calibid, dtype)

        self.bpm = badpixelmask
        self.hwin = hwin
        self.wwin = wwin

    def run(self, img):
        """Replace the bad pixels of the primary data by the median of their neighbours."""
        import numina.array.bpm as bpm

        imgid = self.get_imgid(img)

        _logger.debug("correcting bad pixel mask in %s", imgid)

        data = self.datamodel.get_data(img)
        newdata = bpm.process_bpm_median(data, self.bpm, hwin=self.hwin, wwin=self.wwin, fill=0, reuse_values=True)
        # newdata = array.fixpix(data, self.bpm)
        # FIXME: this breaks datamodel abstraction
        img["primary"].data = newdata
        hdr = img["primary"].header
        hdr["NUM-BPM"] = self.calibid
        hdr["history"] = f"BPM correction with {self.calibid}"
        t_str = datetime.datetime.now(datetime.UTC).isoformat()
        hdr["history"] = f"BPM correction time {t_str}"
        return img


class BiasCorrector(Corrector):
    """A Node that corrects a frame from bias.

    Parameters
    ----------
    biasmap : numpy.ndarray
        Bias image subtracted from the data.
    biasvar : numpy.ndarray, optional
        Variance of the bias image. If given, it is added to the variance of the frame.
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data of the frame.
    calibid : str
        Identifier of the bias, recorded in the header of the corrected frame.
    dtype : str
        Data type to which integer images are promoted, float images keep their type.
    """

    def __init__(self, biasmap, biasvar=None, datamodel=None, calibid="calibid-unknown", dtype="float32"):

        self.update_variance = biasvar is not None

        super().__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)
        self.bias_stats = biasmap.mean()
        self.biasmap = biasmap
        self.biasvar = biasvar

    def run(self, img):
        """Subtract the bias from the data and, if ``biasvar`` was given, update the variance."""

        imgid = self.get_imgid(img)

        _logger.debug("correcting bias in %s", imgid)
        _logger.debug("bias mean is %f", self.bias_stats)

        data = self.datamodel.get_data(img)
        data -= self.biasmap
        # FIXME
        img[0].data = data

        if self.update_variance:
            _logger.debug("update variance with bias")
            variance = self.datamodel.get_variance(img)
            variance += self.biasvar
            img["variance"].data = variance
        hdr = img["primary"].header
        hdr["NUM-BS"] = self.calibid
        hdr["history"] = f"Bias correction with {self.calibid}"
        hdr["history"] = f"Bias image mean is {self.bias_stats}"
        t_str = datetime.datetime.now(datetime.UTC).isoformat()
        hdr["history"] = f"Bias correction time {t_str}"
        return img


class DarkCorrector(Corrector):
    """A Node that corrects a frame from dark current.

    Parameters
    ----------
    darkmap : numpy.ndarray
        Dark current per unit of time, scaled by the dark time of the frame.
    darkvar : numpy.ndarray, optional
        Variance of the dark current. If given, it is scaled by the square
        of the dark time and added to the variance of the frame.
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data and the dark time of the frame.
    calibid : str
        Identifier of the dark, recorded in the header of the corrected frame.
    dtype : str
        Data type to which integer images are promoted, float images keep their type.
    """

    def __init__(self, darkmap, darkvar=None, datamodel=None, calibid="calibid-unknown", dtype="float32"):

        self.update_variance = darkvar is not None

        super().__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)

        self.dark_stats = darkmap.mean()
        self.darkmap = darkmap
        self.darkvar = darkvar

    def run(self, img):
        """Subtract the dark current scaled by the dark time of the frame."""

        etime = self.datamodel.get_darktime(img)

        data = self.datamodel.get_data(img)

        data -= self.darkmap * etime
        # FIXME
        img[0].data = data

        if self.update_variance:
            variance = self.datamodel.get_variance(img)
            variance += self.darkvar * etime * etime
            img["variance"].data = variance
        hdr = img["primary"].header
        hdr["NUM-DK"] = self.calibid
        hdr["history"] = f"Dark correction with {self.calibid}"
        t_str = datetime.datetime.now(datetime.UTC).isoformat()
        hdr["history"] = f"Dark correction time {t_str}"
        return img


class NonLinearityCorrector(Corrector):
    """A Node that corrects a frame from non-linearity.

    Parameters
    ----------
    polynomial : array_like
        Coefficients of the correction polynomial, in the order used by ``numpy.polyval``.
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data of the frame.
    calibid : str
        Identifier of the calibration, recorded in the header of the corrected frame.
    dtype : str
        Data type of the corrected data.
    """

    def __init__(self, polynomial, datamodel=None, calibid="calibid-unknown", dtype="float32"):

        self.update_variance = False

        super().__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)

        self.polynomial = polynomial

    def run(self, img):
        """Replace the data by the correction polynomial evaluated on the data."""
        imgid = self.get_imgid(img)
        _logger.debug("correcting non linearity in %s", imgid)

        data = self.datamodel.get_data(img)

        data = array.correct_nonlinearity(data, self.polynomial, dtype=self.dtype)
        # FIXME
        img[0].data = data
        hdr = self.datamodel.get_header(img)
        hdr["NUM-LIN"] = self.calibid
        hdr["history"] = f"Non-linearity correction with {self.calibid}"
        t_str = datetime.datetime.now(datetime.UTC).isoformat()
        hdr["history"] = f"Non-linearity correction time {t_str}"
        return img


class FlatFieldCorrector(Corrector):
    """A Node that corrects a frame from flat-field.

    Parameters
    ----------
    flatdata : numpy.ndarray
        Flat-field image that divides the data.
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data of the frame.
    calibid : str
        Identifier of the flat-field, recorded in the header of the corrected frame.
    dtype : str
        Data type of the corrected data.
    """

    def __init__(self, flatdata, datamodel=None, calibid="calibid-unknown", dtype="float32"):

        self.update_variance = False

        super().__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)

        self.flatdata = flatdata
        self.flat_stats = flatdata.mean()

    def run(self, img):
        """Divide the data by the flat-field."""
        imgid = self.get_imgid(img)

        _logger.debug("correcting flat in %s", imgid)
        _logger.debug("flat mean is %f", self.flat_stats)

        data = self.datamodel.get_data(img)
        data = array.correct_flatfield(data, self.flatdata, dtype=self.dtype)
        # FIXME
        img[0].data = data
        hdr = img["primary"].header
        hdr["NUM-FF"] = self.calibid
        hdr["history"] = f"Flat-field correction with {self.calibid}"
        t_str = datetime.datetime.now(datetime.UTC).isoformat()
        hdr["history"] = f"Flat-field correction time {t_str}"
        hdr["history"] = f"Flat-field correction mean {self.flat_stats}"
        return img


class SkyCorrector(Corrector):
    """A Node that corrects a frame from sky.

    Parameters
    ----------
    skydata : numpy.ndarray
        Sky image subtracted from the data.
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the data of the frame and to decide
        whether the sky is subtracted.
    calibid : str
        Identifier of the sky, recorded in the header of the corrected frame.
    dtype : str
        Data type of the corrected data.
    """

    def __init__(self, skydata, datamodel=None, calibid="calibid-unknown", dtype="float32"):

        self.update_variance = False

        super().__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)

        self.skydata = skydata
        self.calib_stats = skydata.mean()

    def run(self, img):
        """Subtract the sky, if ``datamodel.do_sky_correction`` returns True for the frame."""
        imgid = self.get_imgid(img)

        if self.datamodel.do_sky_correction(img):
            _logger.debug("correcting sky in %s", imgid)
            _logger.debug("sky mean is %f", self.calib_stats)

            data = self.datamodel.get_data(img)

            data = array.correct_sky(data, self.skydata, dtype=self.dtype)

            # FIXME
            img[0].data = data
            hdr = img["primary"].header
            hdr["NUM-SK"] = self.calibid
            hdr["history"] = f"Sky subtraction with {self.calibid}"
            t_str = datetime.datetime.now(datetime.UTC).isoformat()
            hdr["history"] = f"Sky subtraction time {t_str}"
            hdr["history"] = f"Sky subtraction mean {self.calib_stats}"
        else:
            _logger.debug("skip sky correction in %s", imgid)
        return img


class DivideByExposure(Corrector):
    """A Node that divides its input by exposure time.

    Parameters
    ----------
    datamodel : numina.datamodel.DataModel, optional
        Data model used to access the header and the exposure time of the frame.
    calibid : str
        Not used by this node.
    dtype : str
        Data type to which integer images are promoted.
    """

    def __init__(self, datamodel=None, calibid="calibid-unknown", dtype="float32"):

        self.update_variance = False

        super().__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)

    def run(self, img):
        """Divide the data (and the variance, if present) by the exposure time when BUNIT is ADU.

        Frames with BUNIT equal to ADU/s, without BUNIT or with other values are not changed.
        """
        imgid = self.get_imgid(img)
        hdr = self.datamodel.get_header(img)
        bunit = hdr.get("BUNIT")
        convert_to_s = False
        if bunit:
            if bunit.lower() == "adu":
                convert_to_s = True
            elif bunit.lower() == "adu/s":
                convert_to_s = False
            else:
                _logger.warning("Unrecognized value for BUNIT %s", bunit)
        if convert_to_s:
            etime = self.datamodel.get_exptime(img)
            _logger.debug("divide %s by exposure time %f", imgid, etime)

            img[0].data /= etime
            img[0].header["BUNIT"] = "ADU/s"
            try:
                img["variance"].data /= etime**2
                img["variance"].header["BUNIT"] = "ADU2/s2"
            except KeyError:
                pass

            hdr["NUM-EXP"] = etime
            hdr["history"] = f"Divided by exposure {etime}"
            t_str = datetime.datetime.now(datetime.UTC).isoformat()
            hdr["history"] = f"Divided by exposure {t_str}"
        return img
