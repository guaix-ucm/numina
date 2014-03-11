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

import logging
import time

from .node import Node
import numina.array as array

_logger = logging.getLogger('numina.processing')

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
        header.update(self.tag, time.asctime(), self.comment)

class Corrector(Node):
    '''A Node that corrects a frame from instrumental signatures.'''
    def __init__(self, datamodel, tagger, dtype='float32'):
        super(Corrector, self).__init__()
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
            self._run(img)
            self.tagger.tag_as_processed(hdr)
        return img
    
class TagOptionalCorrector(Corrector):
    def __init__(self, datamodel, tagger, mark=True, dtype='float32'):
        if not mark:
            tagger = NoTag()
        
        super(TagOptionalCorrector, self).__init__(datamodel=datamodel, 
                                                   tagger=tagger, dtype=dtype)

class BadPixelCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from bad pixels.'''
    def __init__(self, badpixelmask, mark=True, tagger=None, datamodel=None, dtype='float32'):
        
        if tagger is None:
            tagger = TagFits('NUM-BPM','Badpixel removed with Numina')
        
        super(BadPixelCorrector, self).__init__(datamodel, tagger, mark, dtype=dtype)
        self.bpm = badpixelmask

    def _run(self, img):
        _logger.debug('correcting bad pixel mask in %s', img)        
        img.data = array.fixpix(img.data, self.bpm)
        return img


class BiasCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from bias.'''
    def __init__(self, biasmap, biasvar=None, datamodel=None, mark=True, 
                 tagger=None, dtype='float32'):
        
        
        
        if tagger is None:
            tagger = TagFits('NUM-BS','Bias removed with Numina')
            
        self.update_variance = False
        
        if biasvar:
            self.update_variance = True
        
        super(BiasCorrector, self).__init__(datamodel=datamodel,
                                            tagger=tagger, 
                                            mark=mark, 
                                            dtype=dtype)
        self.biasmap = biasmap
        self.biasvar = biasvar

    def _run(self, img):
        _logger.debug('correcting bias in %s', img)
        data = self.datamodel.get_data(img)

        
        data -= self.biasmap
        
        if self.update_variance:
            variance = self.datamodel.get_variance(img)
            variance += self.biasvar 
        
        return img

class DarkCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from dark current.'''
    def __init__(self, darkmap, darkvar=None, scale=False, datamodel=None, 
                mark=True, tagger=None, dtype='float32'):
                
        if tagger is None:
            tagger = TagFits('NUM-DK','Dark removed with Numina')
            
        self.update_variance = False
        
        if darkvar:
            self.update_variance = True

        self.scale = scale
        
        super(DarkCorrector, self).__init__(datamodel=datamodel,
                                            tagger=tagger, 
                                            mark=mark, 
                                            dtype=dtype)
        self.darkmap = darkmap
        self.darkvar = darkvar
    
    def _run(self, img):
        _logger.debug('correcting dark in %s', img)
        etime = 1.0
        if self.scale:
            header = self.datamodel.get_header(img)
            etime = header['EXPTIME']
            _logger.debug('scaling dark by %f', etime)

        data = self.datamodel.get_data(img)
        
        data -= self.darkmap * etime
        
        if self.update_variance:
            variance = self.datamodel.get_variance(img)
            variance += self.darkvar * etime * etime
        
        return img

class NonLinearityCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from non-linearity.'''
    def __init__(self, polynomial, datamodel=None, mark=True, 
                 tagger=None, dtype='float32'):
        
        if tagger is None:
            tagger = TagFits('NUM-LIN','Non-linearity corrected with Numina')
            
        self.update_variance = False
                
        super(NonLinearityCorrector, self).__init__(datamodel=datamodel,
                                            tagger=tagger, 
                                            mark=mark, 
                                            dtype=dtype)
        
        self.polynomial = polynomial
                
    def _run(self, img):
        _logger.debug('correcting non linearity in %s', img)
        
        data = self.datamodel.get_data(img)
        
        data = array.correct_nonlinearity(data, self.polynomial, dtype=self.dtype)
        
        return img
        
class FlatFieldCorrector(TagOptionalCorrector):
    '''A Node that corrects a frame from flat-field.'''
    def __init__(self, flatdata, datamodel=None, mark=True, 
                 tagger=None, dtype='float32'):
        
        if tagger is None:
            tagger = TagFits('NUM-FF','Flat-field removed with Numina')
            
        self.update_variance = False
                
        super(FlatFieldCorrector, self).__init__(
            datamodel=datamodel,
            tagger=tagger, 
            mark=mark, 
            dtype=dtype)
        
        self.flatdata = flatdata
                
    def _run(self, img):
        _logger.debug('correcting flat-field in %s', img)
        
        data = self.datamodel.get_data(img)
        
        data = array.correct_flatfield(data, self.flatdata, dtype=self.dtype)
        
        return img
