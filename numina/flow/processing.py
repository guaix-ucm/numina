#
# Copyright 2008-2014 Universidad Complutense de Madrid
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
import time

from astropy.io import fits

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


class SimpleDataModel(object):
    '''Model of the Data being processed'''

    def get_data(self, img):
        return img['primary'].data

    def get_header(self, img):
        return img['primary'].header

    def get_variance(self, img):
        return img['variance'].data


class NoTag(object):
    def check_if_processed(self, img):
        return False

    def tag_as_processed(self, img):
        pass


class TagFits(object):
    def __init__(self, tag, comment):
        self.tag = tag
        self.comment = comment

    def check_if_processed(self, header):
        return self.tag in header

    def tag_as_processed(self, header):
        header[self.tag]= (time.asctime(), self.comment)


class Corrector(Node):
    '''A Node that corrects a frame from instrumental signatures.'''

    def __init__(self, datamodel, tagger=None, dtype='float32'):
        super(Corrector, self).__init__()
        if tagger is None:
            self.tagger = NoTag()
        else:
            self.tagger = tagger
        if not datamodel:
            self.datamodel = SimpleDataModel()
        else:
            self.datamodel = datamodel
        self.dtype = dtype

    def __call__(self, img):
        hdr = self.datamodel.get_header(img)
        if self.tagger.check_if_processed(hdr):
            _logger.info('%s already processed by %s', img, self)
            return img
        else:
            if img[0].data.dtype in ['<u2', '>u2', '=u2']:
                # FIXME
                _logger.info('change dtype to float32, old is %s',
                             img[0].data.dtype)
                img = promote_hdulist(img)
            img = self._run(img)
            self.tagger.tag_as_processed(hdr)
        return img

    def get_imgid(self, img):

        imgid = img.filename()

        # More heuristics here...
        # get FILENAME keyword, for example...

        if not imgid:
            imgid = repr(img)

        return imgid


class TagOptionalCorrector(Corrector):
    def __init__(self, datamodel, tagger, mark=True, dtype='float32'):
        if not mark:
            tagger = NoTag()

        super(TagOptionalCorrector, self).__init__(datamodel=datamodel,
                                                   tagger=tagger, dtype=dtype)


class BadPixelCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from bad pixels.'''

    def __init__(self, badpixelmask, mark=True, tagger=None,
                 datamodel=None, dtype='float32'):
        if tagger is None:
            tagger = TagFits('NUM-BPM', 'Badpixel removed with Numina')

        super(BadPixelCorrector, self).__init__(datamodel, tagger, mark, dtype)

        self.bpm = badpixelmask

    def _run(self, img):
        # import numpy
        import scipy.signal
        imgid = self.get_imgid(img)

        _logger.debug('correcting bad pixel mask in %s', imgid)

        # data = self.datamodel.get_data(img)
        # newdata = array.fixpix(data, self.bpm)
        # FIXME: this breaks datamodel abstraction
        # img[0].data = newdata
        # img[0].data[self.bpm.astype(bool)] = numpy.median(img[0].data[~self.bpm.astype(bool)])
        img_sm = scipy.signal.medfilt(img[0].data, 3)
        img[0].data[self.bpm.astype(bool)] = img_sm[self.bpm.astype(bool)]
        # img[0].data[self.bpm.astype(bool)] = 46
        return img


class BiasCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from bias.'''

    def __init__(self, biasmap, biasvar=None, datamodel=None, mark=True,
                 tagger=None, dtype='float32'):

        if tagger is None:
            tagger = TagFits('NUM-BS','Bias removed with Numina')

        self.update_variance = True if biasvar else False

        super(BiasCorrector, self).__init__(datamodel=datamodel,
                                            tagger=tagger,
                                            mark=mark,
                                            dtype=dtype)
        self.bias_stats = biasmap.mean()
        self.biasmap = biasmap
        self.biasvar = biasvar

    def _run(self, img):

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

        return img


class DarkCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from dark current.'''

    def __init__(self, darkmap, darkvar=None, datamodel=None,
                 mark=True, tagger=None, dtype='float32'):

        if tagger is None:
            tagger = TagFits('NUM-DK',
                             'Dark removed with Numina')

        self.update_variance = False

        if darkvar:
            self.update_variance = True

        super(DarkCorrector, self).__init__(datamodel=datamodel,
                                            tagger=tagger,
                                            mark=mark,
                                            dtype=dtype)

        self.dark_stats = darkmap.mean()
        self.darkmap = darkmap
        self.darkvar = darkvar

    def _run(self, img):

        header = self.datamodel.get_header(img)
        if 'EXPTIME' in header.keys():
            etime = header['EXPTIME']
        elif 'EXPOSED' in header.keys():
            etime = header['EXPOSED']
        else:
            etime = 1.0

        data = self.datamodel.get_data(img)

        data -= self.darkmap * etime
        # FIXME
        img[0].data = data

        if self.update_variance:
            variance = self.datamodel.get_variance(img)
            variance += self.darkvar * etime * etime
            # FIXME
            img[1].data = variance

        return img


class NonLinearityCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from non-linearity.'''

    def __init__(self, polynomial, datamodel=None, mark=True,
                 tagger=None, dtype='float32'):
        if tagger is None:
            tagger = TagFits('NUM-LIN',
                             'Non-linearity corrected with Numina')

        self.update_variance = False

        super(NonLinearityCorrector, self).__init__(
            datamodel=datamodel, tagger=tagger,
            mark=mark, dtype=dtype
        )

        self.polynomial = polynomial

    def _run(self, img):
        _logger.debug('correcting non linearity in %s', img)

        data = self.datamodel.get_data(img)

        data = array.correct_nonlinearity(data, self.polynomial,
                                          dtype=self.dtype)
        # FIXME
        img[0].data = data
        return img


class FlatFieldCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from flat-field.'''

    def __init__(self, flatdata, datamodel=None, mark=True,
                 tagger=None, dtype='float32'):
        if tagger is None:
            tagger = TagFits('NUM-FF', 'Flat-field removed with Numina')

        self.update_variance = False

        super(FlatFieldCorrector, self).__init__(
            datamodel=datamodel,
            tagger=tagger,
            mark=mark,
            dtype=dtype)

        self.flatdata = flatdata
        self.flat_stats = flatdata.mean()

    def _run(self, img):
        imgid = self.get_imgid(img)

        _logger.debug('correcting flat in %s', imgid)
        _logger.debug('flat mean is %f', self.flat_stats)

        data = self.datamodel.get_data(img)
        data = array.correct_flatfield(data, self.flatdata, dtype=self.dtype)
        # FIXME
        img[0].data = data
        return img


class SkyCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from sky.'''

    def __init__(self, skydata, datamodel=None, mark=True,
                 tagger=None, dtype='float32'):
        if tagger is None:
            tagger = TagFits('NUM-SK', 'Sky removed with Numina')

        self.update_variance = False

        super(SkyCorrector, self).__init__(
            datamodel=datamodel,
            tagger=tagger,
            mark=mark,
            dtype=dtype)

        self.skydata = skydata
        self.calib_stats = skydata.mean()

    def _run(self, img):
        imgid = self.get_imgid(img)

        _logger.debug('correcting sky in %s', imgid)
        _logger.debug('sky mean is %f', self.calib_stats)

        data = self.datamodel.get_data(img)

        data = array.correct_sky(data, self.skydata, dtype=self.dtype)

        # FIXME
        img[0].data = data

        return img


class DivideByExposure(TagOptionalCorrector):
    '''A Node that divides its input by exposure time.'''

    def __init__(self, datamodel=None, mark=True,
                 tagger=None, dtype='float32'):

        if tagger is None:
            tagger = TagFits('NUM-EXP', 'Divided by exposure time.')

        self.update_variance = False

        super(DivideByExposure, self).__init__(
            datamodel=datamodel,
            tagger=tagger,
            mark=mark,
            dtype=dtype
        )

    def _run(self, img):
        imgid = self.get_imgid(img)
        header = self.datamodel.get_header(img)
        bunit = header['BUNIT']
        convert_to_s = False
        if bunit:
            if bunit.lower() == 'adu':
                convert_to_s = True
            elif bunit.lower() == 'adu/s':
                convert_to_s = False
            else:
                _logger.warning('Unrecognized value for BUNIT %s', bunit)
        if convert_to_s:
            etime = header['EXPTIME']
            _logger.debug('divide %s by exposure time %f', imgid, etime)

            img[0].data /= etime
            img[0].header['BUNIT'] = 'ADU/s'
            try:
                img['variance'].data /= etime ** 2
                img['variance'].header['BUNIT'] = 'ADU/s'
            except KeyError:
                pass

        return img
