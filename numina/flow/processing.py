#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

import logging

from .node import Corrector
import numina.array as array

_logger = logging.getLogger('numina.processing')

class BadPixelCorrector(Corrector):
    def __init__(self, badpixelmask, mark=True, dtype='float32'):
        super(BadPixelCorrector, self).__init__(label=('NUM-BPM','Badpixel removed with numina'), dtype=dtype, mark=mark)
        self.bpm = badpixelmask

    def _run(self, img):
        _logger.debug('correcting bad pixel mask in %s', img)        
        img.data = array.fixpix(img.data, self.bpm)
        self.mark_as_processed(img)
        return img


class BiasCorrector(Corrector):
    def __init__(self, biasmap, mark=True, dtype='float32'):
        super(BiasCorrector, self).__init__(label=('NUM-BS','Bias removed with numina'), dtype=dtype, mark=mark)
        self.biasmap = biasmap

    def _run(self, img):
        _logger.debug('correcting bias in %s', img)
        img.data = array.correct_dark(img.data, self.biasmap, dtype=self.dtype)
        self.mark_as_processed(img)
        return img

class DarkCorrector(Corrector):
    def __init__(self, darkmap, mark=True, dtype='float32'):
        super(DarkCorrector, self).__init__(label=('NUM-DK','Dark removed with numina'), dtype=dtype, mark=mark)
        self.darkmap = darkmap

    def _run(self, img):
        _logger.debug('correcting dark in %s', img)
        img.data = array.correct_dark(img.data, self.darkmap, dtype=self.dtype)
        self.mark_as_processed(img)
        return img

class NonLinearityCorrector(Corrector):
    def __init__(self, polynomial, mark=True, dtype='float32'):
        super(NonLinearityCorrector, self).__init__(label=('NUM-LIN','Non-linearity removed with numina'), dtype=dtype, mark=mark)
        self.polynomial = polynomial
                
    def _run(self, img):
        _logger.debug('correcting non linearity in %s', img)
        img.data = array.correct_nonlinearity(img.data, self.polynomial, dtype=self.dtype)
        self.mark_as_processed(img)
        return img
        
class FlatFieldCorrector(Corrector):
    def __init__(self, flatdata, mark=True, dtype='float32'):
        super(FlatFieldCorrector, self).__init__(label=('NUM-FF','Flat field removed with numina'), dtype=dtype, mark=mark)
        self.flatdata = flatdata

    def _run(self, img):
        _logger.debug('correcting flatfield in %s', img)
        img.data = array.correct_flatfield(img.data, self.flatdata, dtype=self.dtype)
        self.mark_as_processed(img)
        return img

