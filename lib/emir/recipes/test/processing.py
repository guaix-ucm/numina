
import logging
import copy

import numpy

import node
from numina.array import subarray_match

_logger = logging.getLogger('numina.processing')

def version_file(file_spec, vtype='copy'):
    import os, shutil
    if os.path.isfile(file_spec):
        if vtype not in ('copy', 'move'):
            raise ValueError, 'Unknown vtype %r', (vtype, )
        n, e = os.path.splitext(file_spec)
        if len(e) == 4 and e[1:].isdigit():
            num = 1 + int(e[1:])
            root = n
        else:
            num = 0
            root = file_spec
        for i in xrange(num, 1000):
            new_file = '%s.%03d' % (root, i)
            if not os.path.exists(new_file):
                if vtype == 'copy':
                    shutil.copy(file_spec, new_file)
                else:
                    os.rename(file_spec, new_file)
                return True
        raise RuntimeError("Can't %s %r, all names taken" % (vtype, file_spec))
    return False

class OpenNode(node.Node):
    def __init__(self, mode='copyonwrite', memmap=False):
        super(OpenNode, self).__init__()
        self.mode = mode
        self.memmap = memmap

    def __call__(self, ri):
        _logger.debug('opening %s', ri)
        ri.open(mode=self.mode, memmap=self.memmap)
        return ri

class SaveAsNode(node.Node):
    
    def uuidname(self, img_):
        import uuid
        d = uuid.uuid4().hex
        m = uuid.uuid4().hex
        return (d, m)
    
    def __init__(self, namegen=None):
        super(SaveAsNode, self).__init__()
        if namegen is None:
            self.namegen = self.uuidname
        else:
            self.namegen = namegen
             
    def check_if_processed(self, img):
        return False
    
    def __call__(self, img):
        names = self.namegen(img)
        img.hdulist[0].writeto(names[0], output_verify='ignore', clobber=True)
        img.hdulist[1].writeto(names[1], output_verify='ignore', clobber=True)
        #newimg = image.EmirImage(names[0], names[1], offset=img.offset)
        newimg = copy.copy(img)
        newimg.datafile = names[0]
        newimg.maskfile = names[1]
        newimg.close()
        return img, newimg

class CloseNode(node.Node):
    def __init__(self, output_verify='exception'):
        super(CloseNode, self).__init__()
        self.output_verify = output_verify

    def __call__(self, ri):
        _logger.debug('closing %s', ri)
        ri.close(output_verify=self.output_verify)
        return ri

class BackupNode(node.Node):
    def __init__(self):
        super(BackupNode, self).__init__()

    def __call__(self, ri):
        _logger.debug('backup %s', ri)
        version_file(ri.datafile)
        return ri

class ResizeNode(node.Corrector):
    def __init__(self, finalshape):
        super(ResizeNode, self).__init__()
        self.finals = finalshape
  
    def __call__(self, img):
        newdata = numpy.zeros(self.finals, dtype=img.data.dtype)
        region, ign_ = subarray_match(self.finals, img.noffset, img.data.shape)
        assert newdata[region].shape == img.data.shape
        newdata[region] = img.data
        img.region = region
        img.data = newdata
        return img
    
    def check_if_processed(self, img):
        return False

class BiasCorrector(node.Corrector):
    def __init__(self, biasmap, mark=True, dtype='float32'):
        super(BiasCorrector, self).__init__(label=('NUM-DK','Dark removed with numina'), dtype=dtype, mark=mark)
        self.biasmap = biasmap

    def __call__(self, image):
        _logger.debug('correcting bias in %s', image)
        image.data -= self.biasmap
        image.data = image.data.astype(self.dtype)
        return image

class DarkCorrector(node.Corrector):
    def __init__(self, darkmap, mark=True, dtype='float32'):
        super(DarkCorrector, self).__init__(label=('NUM-DK','Bias removed with numina'), dtype=dtype, mark=mark)
        self.darkmap = darkmap

    def __call__(self, image):
        _logger.debug('correcting dark in %s', image)
        image.data -= self.darkmap
        image.data = image.data.astype(self.dtype)
        return image

class NonLinearityCorrector(node.Corrector):
    def __init__(self, polynomial, mark=True, dtype='float32'):
        super(NonLinearityCorrector, self).__init__(label=('NUM-LIN','Non-linearity removed with numina'), dtype=dtype, mark=mark)
        self.polynomial = polynomial
                
    def __call__(self, image):
        _logger.debug('correcting non linearity in %s', image)
        image.data = numpy.polyval(self.polynomial, image.data)
        image.data = image.data.astype(self.dtype)
        return image
        
class FlatFieldCorrector(node.Corrector):
    def __init__(self, flatdata, mark=True, dtype='float32'):
        super(FlatFieldCorrector, self).__init__(label=('NUM-FF','Flat field removed with numina'), dtype=dtype, mark=mark)
        self.flatdata = flatdata

    def __call__(self, image):
        _logger.debug('correcting flatfield in %s', image)
        image.data /= self.flatdata
        image.data = image.data.astype(self.dtype)
        return image

def compute_median(img):
    value = numpy.median(img.data[img.mask == 0])
    _logger.debug('median value of %s is %f', img, value)
    return value, img
