
import pyfits
import numpy

from numina.array.combine import median

class ObservingMode(object):
    pass

class ObservingBlock(object):
    def __init__(self, images):
        self.images
        self.dither_pattern

class Calibration(object):
    pass


class FitsHandle(object):
    def __init__(self):
        pass


class EmirImage(object):
    def __init__(self, datafile, maskfile):
        super(EmirImage, self).__init__()
        self.data = None
        self.mask = None
        self.region = None
        self.proc = False
        self.meta = None
        self.datafile = datafile
        self.maskfile = maskfile
        self._open = False
        self.hdulist = []

    def is_open(self):
        return self._open

    def open(self, mode='copyonwrite', memmap=False):
        self._open = True
        self.hdulist.append(pyfits.open(self.datafile, memmap=memmap, mode=mode))
        self.data = self.hdulist[0]['primary'].data
        self.meta = self.hdulist[0]['primary'].header
        self.hdulist.append(pyfits.open(self.maskfile, memmap=memmap, mode=mode))
        self.mask = self.hdulist[-1]['primary'].data

    def close(self):
        if self.is_open():
            for h in self.hdulist:
                h.flush()
                h.close()
        self._open = False
        self.data = None
        self.mask = None

    def __str__(self):
        return 'EmirImage(datafile=%s, maskfile=%s)' % (self.datafile, self.maskfile)

def combine(mode, images, scales=None, *args, **kwds):
    data = [i.data for i in images]
    masks = [i.mask for i in images]
    
    superflat = median(data, masks, scales=scales)

    return superflat
