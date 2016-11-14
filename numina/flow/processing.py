#
# Copyright 2008-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function

import logging
import warnings
import datetime

from astropy.io import fits

from .datamodel import DataModel
from .node import Node
import numina.array as array

_logger = logging.getLogger('numina.processing')


def promote_hdulist(hdulist, totype='float32'):
    nn = [promote_hdu(hdu, totype=totype) for hdu in hdulist]
    newhdulist = fits.HDUList(nn)
    return newhdulist


def promote_hdu(hdu, totype='float32'):
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


class SimpleDataModel(DataModel):
    """Model of the Data being processed"""
    pass


class Corrector(Node):
    """A Node that corrects a frame from instrumental signatures."""

    def __init__(self, datamodel=None, calibid='calibid-unknown', dtype='float32'):
        super(Corrector, self).__init__()
        if not datamodel:
            self.datamodel = SimpleDataModel()
        else:
            self.datamodel = datamodel
        self.dtype = dtype
        self.calibid = calibid

    def __call__(self, img):
        if img[0].data.dtype in ['<u2', '>u2', '=u2']:
            # FIXME: this is a GCS problem
            _logger.info('change dtype to float32, old is %s',
                         img[0].data.dtype)
            img = promote_hdulist(img)
        if hasattr(self, 'run'):
            img = self.run(img)
        else:
            warnings.warn("use method 'run' instead of '_run'", DeprecationWarning)
            img = self.run(img)

        return img

    def get_imgid(self, img):
        return self.datamodel.get_imgid(img)


TagOptionalCorrector = Corrector


class BadPixelCorrector(Corrector):
    """A Node that corrects a frame from bad pixels."""

    def __init__(self, badpixelmask, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        super(BadPixelCorrector, self).__init__(datamodel, calibid, dtype)

        self.bpm = badpixelmask

    def run(self, img):
        import numina.array.bpm as bpm
        imgid = self.get_imgid(img)

        _logger.debug('correcting bad pixel mask in %s', imgid)

        data = self.datamodel.get_data(img)
        newdata = bpm.process_bpm_median(data, self.bpm, hwin=2, wwin=2, fill=0)
        # newdata = array.fixpix(data, self.bpm)
        # FIXME: this breaks datamodel abstraction
        img['primary'].data = newdata
        hdr = img['primary'].header
        hdr['NUM-BPM'] = self.calibid
        hdr['history'] = 'BPM correction with {}'.format(self.calibid)
        hdr['history'] = 'BPM correction time {}'.format(datetime.datetime.utcnow().isoformat())
        return img


class BiasCorrector(Corrector):
    """A Node that corrects a frame from bias."""

    def __init__(self, biasmap, biasvar=None, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        self.update_variance = True if biasvar else False

        super(BiasCorrector, self).__init__(datamodel=datamodel, calibid=calibid, dtype=dtype)
        self.bias_stats = biasmap.mean()
        self.biasmap = biasmap
        self.biasvar = biasvar

    def run(self, img):

        imgid = self.get_imgid(img)

        _logger.debug('correcting bias in %s', imgid)
        _logger.debug('bias mean is %f', self.bias_stats)

        data = self.datamodel.get_data(img)
        data -= self.biasmap
        # FIXME
        img[0].data = data

        if self.update_variance:
            _logger.debug('update variance with bias')
            variance = self.datamodel.get_variance(img)
            variance += self.biasvar
            # FIXME
            img[1].data = variance
        hdr = img['primary'].header
        hdr['NUM-BS'] = self.calibid
        hdr['history'] = 'Bias correction with {}'.format(self.calibid)
        hdr['history'] = 'Bias image mean is {}'.format(self.bias_stats)
        hdr['history'] = 'Bias correction time {}'.format(datetime.datetime.utcnow().isoformat())
        return img


class DarkCorrector(Corrector):
    """A Node that corrects a frame from dark current."""

    def __init__(self, darkmap, darkvar=None, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        self.update_variance = False

        if darkvar:
            self.update_variance = True

        super(DarkCorrector, self).__init__(datamodel=datamodel,
                                            calibid=calibid,
                                            dtype=dtype)

        self.dark_stats = darkmap.mean()
        self.darkmap = darkmap
        self.darkvar = darkvar

    def run(self, img):

        etime = self.datamodel.get_darktime(img)

        data = self.datamodel.get_data(img)

        data -= self.darkmap * etime
        # FIXME
        img[0].data = data

        if self.update_variance:
            variance = self.datamodel.get_variance(img)
            variance += self.darkvar * etime * etime
            # FIXME
            img[1].data = variance
        hdr = img['primary'].header
        hdr['NUM-DK'] = self.calibid
        hdr['history'] = 'Dark correction with {}'.format(self.calibid)
        hdr['history'] = 'Dark correction time {}'.format(datetime.datetime.utcnow().isoformat())
        return img


class NonLinearityCorrector(Corrector):
    """A Node that corrects a frame from non-linearity."""

    def __init__(self, polynomial, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        self.update_variance = False

        super(NonLinearityCorrector, self).__init__(
            datamodel=datamodel, calibid=calibid, dtype=dtype
        )

        self.polynomial = polynomial

    def run(self, img):
        _logger.debug('correcting non linearity in %s', img)

        data = self.datamodel.get_data(img)

        data = array.correct_nonlinearity(data, self.polynomial,
                                          dtype=self.dtype)
        # FIXME
        img[0].data = data
        hdr = self.datamodel.get_header(img)
        hdr['NUM-LIN'] = self.calibid
        hdr['history'] = 'Non-linearity correction with {}'.format(self.calibid)
        hdr['history'] = 'Non-linearity correction time {}'.format(datetime.datetime.utcnow().isoformat())
        return img


class FlatFieldCorrector(Corrector):
    """A Node that corrects a frame from flat-field."""

    def __init__(self, flatdata, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        self.update_variance = False

        super(FlatFieldCorrector, self).__init__(
            datamodel=datamodel,
            calibid=calibid,
            dtype=dtype)

        self.flatdata = flatdata
        self.flat_stats = flatdata.mean()

    def run(self, img):
        imgid = self.get_imgid(img)

        _logger.debug('correcting flat in %s', imgid)
        _logger.debug('flat mean is %f', self.flat_stats)

        data = self.datamodel.get_data(img)
        data = array.correct_flatfield(data, self.flatdata, dtype=self.dtype)
        # FIXME
        img[0].data = data
        hdr = img['primary'].header
        hdr['NUM-FF'] = self.calibid
        hdr['history'] = 'Flat-field correction with {}'.format(self.calibid)
        hdr['history'] = 'Flat-field correction time {}'.format(datetime.datetime.utcnow().isoformat())
        hdr['history'] = 'Flat-field correction mean {}'.format(self.flat_stats)
        return img


class SkyCorrector(Corrector):
    """A Node that corrects a frame from sky."""

    def __init__(self, skydata, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        self.update_variance = False

        super(SkyCorrector, self).__init__(
            datamodel=datamodel,
            calibid=calibid,
            dtype=dtype)

        self.skydata = skydata
        self.calib_stats = skydata.mean()

    def run(self, img):
        imgid = self.get_imgid(img)

        if self.datamodel.do_sky_correction(img):
            _logger.debug('correcting sky in %s', imgid)
            _logger.debug('sky mean is %f', self.calib_stats)

            data = self.datamodel.get_data(img)

            data = array.correct_sky(data, self.skydata, dtype=self.dtype)

            # FIXME
            img[0].data = data
            hdr = img['primary'].header
            hdr['NUM-SK'] = self.calibid
            hdr['history'] = 'Sky subtraction with {}'.format(self.calibid)
            hdr['history'] = 'Sky subtraction time {}'.format(datetime.datetime.utcnow().isoformat())
            hdr['history'] = 'Sky subtraction mean {}'.format(self.calib_stats)
        else:
            _logger.debug('skip sky correction in %s', imgid)
        return img


class DivideByExposure(Corrector):
    """A Node that divides its input by exposure time."""

    def __init__(self, datamodel=None, calibid='calibid-unknown', dtype='float32'):

        self.update_variance = False

        super(DivideByExposure, self).__init__(
            datamodel=datamodel,
            calibid=calibid,
            dtype=dtype
        )

    def run(self, img):
        imgid = self.get_imgid(img)
        hdr = self.datamodel.get_header(img)
        bunit = hdr['BUNIT']
        convert_to_s = False
        if bunit:
            if bunit.lower() == 'adu':
                convert_to_s = True
            elif bunit.lower() == 'adu/s':
                convert_to_s = False
            else:
                _logger.warning('Unrecognized value for BUNIT %s', bunit)
        if convert_to_s:
            etime = self.datamodel.get_exptime(img)
            _logger.debug('divide %s by exposure time %f', imgid, etime)

            img[0].data /= etime
            img[0].header['BUNIT'] = 'ADU/s'
            try:
                img['variance'].data /= etime ** 2
                img['variance'].header['BUNIT'] = 'ADU/s'
            except KeyError:
                pass

            hdr['NUM-EXP'] = etime
            hdr['history'] = 'Divided by exposure {}'.format(etime)
            hdr['history'] = 'Divided by exposure {}'.format(datetime.datetime.utcnow().isoformat())
        return img
