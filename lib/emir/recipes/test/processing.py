
import logging
import copy

import numpy
import pyfits

import node
import image
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

class OpenImage(node.Node):
    def __init__(self, mode='copyonwrite', memmap=False):
        super(OpenImage, self).__init__()
        self.mode = mode
        self.memmap = memmap

    def _run(self, ri):
        _logger.debug('opening %s', ri)
        ri.open(mode=self.mode, memmap=self.memmap)
        return ri

class SaveAsNode(node.Node):
    
    def uuidname(self, _img):
        import uuid
        d = uuid.uuid4().hex
        m = uuid.uuid4().hex
        return (d, m)
    
    def __init__(self, namegen=None):
        super(SaveAsNode, self).__init__(ninputs=1, noutputs=2)
        if namegen is None:
            self.namegen = self.uuidname
        else:
            self.namegen = namegen
    
    def _run(self, img):
        names = self.namegen(img)
        
        img.hdulist[0].writeto(names[0], output_verify='ignore', clobber=True)
        img.hdulist[1].writeto(names[1], output_verify='ignore', clobber=True)
        newimg = copy.copy(img)
        newimg.filename = names[0]
        newimg.maskfile = names[1]
        _logger.debug('saving %s as %s', img, newimg)
        return img, newimg
    
class CopyMask(node.Node):
    
    def uuidname(self, _img):
        import uuid
        d = uuid.uuid4().hex
        m = uuid.uuid4().hex
        return (d, m)
    
    def __init__(self, namegen=None):
        super(CopyMask, self).__init__(ninputs=1, noutputs=2)
        if namegen is None:
            self.namegen = self.uuidname
        else:
            self.namegen = namegen
    
    def _run(self, img):
        names = self.namegen(img)
        
        img.hdulist[1].writeto(names, output_verify='ignore', clobber=True)
        newimg = image.Image(datafile=names)
        newimg.region = img.region
        newimg.label = img.label
        _logger.debug('saving %s as %s', img, newimg)
        return img, newimg

class CloseImage(node.Node):
    def __init__(self, output_verify='exception'):
        super(CloseImage, self).__init__()
        self.output_verify = output_verify

    def _run(self, ri):
        _logger.debug('closing %s', ri)
        ri.close(output_verify=self.output_verify)
        return ri

class BackupNode(node.Node):
    def __init__(self):
        super(BackupNode, self).__init__()

    def _run(self, ri):
        _logger.debug('backup %s', ri)
        version_file(ri.filename)
        return ri

class ResizeNode(node.Corrector):
    def __init__(self, finalshape):
        super(ResizeNode, self).__init__()
        self.finals = finalshape
  
    def _run(self, img):
        newdata = numpy.zeros(self.finals, dtype=_imgs.data.dtype)
        region, _ign = subarray_match(self.finals, img.noffset, _imgs.data.shape)
        
        newdata[region] = img.data
        img.region = region
        img.data = newdata
        
        newmask = numpy.zeros(self.finals, dtype=_imgs.data.dtype)
        newmask[region] = img._masks
        img._masks = newmask
        
        return img

class BiasCorrector(node.Corrector):
    def __init__(self, biasmap, mark=True, dtype='float32'):
        super(BiasCorrector, self).__init__(label=('NUM-DK','Dark removed with numina'), dtype=dtype, mark=mark)
        self.biasmap = biasmap

    def _run(self, img):
        _logger.debug('correcting bias in %s', img)
        img.data -= self.biasmap
        img.data = _imgs.data.astype(self.dtype)
        self.mark_as_processed(img)
        return img

class DarkCorrector(node.Corrector):
    def __init__(self, darkmap, mark=True, dtype='float32'):
        super(DarkCorrector, self).__init__(label=('NUM-DK','Bias removed with numina'), dtype=dtype, mark=mark)
        self.darkmap = darkmap

    def _run(self, img):
        _logger.debug('correcting dark in %s', img)
        img.data -= self.darkmap
        img.data = _imgs.data.astype(self.dtype)
        self.mark_as_processed(img)
        return img

class NonLinearityCorrector(node.Corrector):
    def __init__(self, polynomial, mark=True, dtype='float32'):
        super(NonLinearityCorrector, self).__init__(label=('NUM-LIN','Non-linearity removed with numina'), dtype=dtype, mark=mark)
        self.polynomial = polynomial
                
    def _run(self, img):
        _logger.debug('correcting non linearity in %s', img)
        img.data = numpy.polyval(self.polynomial, img.data)
        img.data = _imgs.data.astype(self.dtype)
        self.mark_as_processed(img)
        return img
        
class FlatFieldCorrector(node.Corrector):
    def __init__(self, flatdata, mark=True, region=False, dtype='float32'):
        super(FlatFieldCorrector, self).__init__(label=('NUM-FF','Flat field removed with numina'), dtype=dtype, mark=mark)
        self.flatdata = flatdata
        self.region = region

    def _run(self, img):
        _logger.debug('correcting flatfield in %s', img)
        if self.region and img.region is not None:
            img.data[img.region] /= self.flatdata
        else:
            img.data /= self.flatdata
        img.data = _imgs.data.astype(self.dtype)
        self.mark_as_processed(img)
        return img

def compute_median(img, mask, region):
    d = img.data[region]
    m = mask.data[region]
    value = numpy.median(d[m == 0])
    _logger.debug('median value of %s is %f', img, value)
    return value, img


class SextractorObjectMask(node.Node):
    def __init__(self, namegen):
        super(SextractorObjectMask, self).__init__()
        self.namegen = namegen
        
    def _run(self, array):
        import tempfile
        import subprocess
        import os.path
    
        checkimage = self.namegen(None)
    
        # A temporary filename used to store the array in fits format
        tf = tempfile.NamedTemporaryFile(prefix='emir_', dir='.')
        pyfits.writeto(filename=tf, data=array)
        
        # Run sextractor, it will create a image called check.fits
        # With the segmentation _masks inside
        sub = subprocess.Popen(["sex",
                                "-CHECKIMAGE_TYPE", "SEGMENTATION",
                                "-CHECKIMAGE_NAME", checkimage,
                                '-VERBOSE_TYPE', 'QUIET',
                                tf.name],
                                stdout=subprocess.PIPE)
        sub.communicate()
    
        segfile = os.path.join('.', checkimage)
    
        # Read the segmentation image
        result = pyfits.getdata(segfile)
    
        # Close the tempfile
        tf.close()    
    
        return result
