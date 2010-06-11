
import os.path

import pyfits

from numina.array.combine import median

class Image(object):
    def __init__(self, datafile):
        super(Image, self).__init__()
        self.data = None
        self.region = None
        self.meta = None
        self.datafile = datafile
        self._open = False
        self.hdulist = [None]
        self.label, _ext = os.path.splitext(datafile)

    def is_open(self):
        return self._open

    def open(self, mode='copyonwrite', memmap=False):
        if not self.is_open():
            self._open = True
            self.hdulist[0] = pyfits.open(self.datafile, memmap=memmap, mode=mode)
            self.data = self.hdulist[0]['primary'].data
            self.meta = self.hdulist[0]['primary'].header
        

    def close(self, output_verify='exception'):
        if self.is_open():
            self.hdulist[0]['primary'].data = self.data
            self.hdulist[0].close(output_verify=output_verify)
        self._open = False
        self.data = None
        self.hdulist = [None]
        
    def __copy__(self):
        new = Image(datafile=self.datafile, maskfile=self.maskfile)
        new.region = self.region
        new.label = self.label
        return new

    def __str__(self):
        return 'Image(datafile="%s")' % (self.datafile)

class EmirImage(object):
    def __init__(self, datafile, maskfile, offset=None):
        super(EmirImage, self).__init__()
        self.data = None
        self.mask = None
        self.offset = offset
        self.noffset = None
        self.region = None
        self.meta = None
        self.datafile = datafile
        self.maskfile = maskfile
        self._open = False
        self.hdulist = [None, None]
        self.label, _ext = os.path.splitext(datafile)

    def is_open(self):
        return self._open

    def open(self, mode='copyonwrite', memmap=False):
        if not self.is_open():
            self._open = True
            self.hdulist[0] = pyfits.open(self.datafile, memmap=memmap, mode=mode)
            self.data = self.hdulist[0]['primary'].data
            self.meta = self.hdulist[0]['primary'].header
            self.hdulist[1] = pyfits.open(self.maskfile, memmap=memmap, mode=mode)
            self.mask = self.hdulist[-1]['primary'].data

    def close(self, output_verify='exception'):
        if self.is_open():
            self.hdulist[0]['primary'].data = self.data
            self.hdulist[0].close(output_verify=output_verify)
            self.hdulist[1]['primary'].data = self.mask
            self.hdulist[1].close(output_verify=output_verify)
        self._open = False
        self.data = None
        self.mask = None
        self.hdulist = [None, None]
        
    def __copy__(self):
        new = EmirImage(datafile=self.datafile, maskfile=self.maskfile)
        new.offset = self.offset
        new.noffset = self.noffset
        new.region = self.region
        new.label = self.label
        return new

    def __str__(self):
        return 'EmirImage(datafile="%s", maskfile="%s")' % (self.datafile, self.maskfile)

def combine(mode, images, *args, **kwds):
    data = [i.data[i.region] for i in images]
    masks = [i.mask[i.region] for i in images]
    superflat = median(data, masks, *args, **kwds)
    return superflat



