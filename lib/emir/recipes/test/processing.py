
import logging

import node
import array

_logger = logging.getLogger('numina.processing')

class BiasCorrector(node.Corrector):
    def __init__(self, biasmap, mark=True, dtype='float32'):
        super(BiasCorrector, self).__init__(label=('NUM-DK','Dark removed with numina'), dtype=dtype, mark=mark)
        self.biasmap = biasmap

    def _run(self, img):
        _logger.debug('correcting bias in %s', img)
        img.data = array.correct_dark(img.data, self.biasmap, dtype=self.dtype)
        self.mark_as_processed(img)
        return img

class DarkCorrector(node.Corrector):
    def __init__(self, darkmap, mark=True, dtype='float32'):
        super(DarkCorrector, self).__init__(label=('NUM-DK','Bias removed with numina'), dtype=dtype, mark=mark)
        self.darkmap = darkmap

    def _run(self, img):
        _logger.debug('correcting dark in %s', img)
        img.data = array.correct_dark(img.data, self.darkmap, dtype=self.dtype)
        self.mark_as_processed(img)
        return img

class NonLinearityCorrector(node.Corrector):
    def __init__(self, polynomial, mark=True, dtype='float32'):
        super(NonLinearityCorrector, self).__init__(label=('NUM-LIN','Non-linearity removed with numina'), dtype=dtype, mark=mark)
        self.polynomial = polynomial
                
    def _run(self, img):
        _logger.debug('correcting non linearity in %s', img)
        img.data = array.correct_nonlinearity(img.data, self.polynomial, dtype=self.dtype)
        self.mark_as_processed(img)
        return img
        
class FlatFieldCorrector(node.Corrector):
    def __init__(self, flatdata, mark=True, dtype='float32'):
        super(FlatFieldCorrector, self).__init__(label=('NUM-FF','Flat field removed with numina'), dtype=dtype, mark=mark)
        self.flatdata = flatdata

    def _run(self, img):
        _logger.debug('correcting flatfield in %s', img)
        img.data = array.correct_flatfield(img.data, self.flatdata, dtype=self.dtype)
        self.mark_as_processed(img)
        return img

