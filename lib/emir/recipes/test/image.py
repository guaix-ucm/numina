
import pyfits

class Image(object):
    def __init__(self, filename):
        super(Image, self).__init__()
        self.data = None
        self.meta = None
        self.filename = filename
        self._open = False
        self.hdulist = None

    def is_open(self):
        return self._open

    def open(self, mode='copyonwrite', memmap=False):
        if not self.is_open():
            self._open = True
            self.hdulist = pyfits.open(self.filename, memmap=memmap, mode=mode)
            self.data = self.hdulist['primary'].data
            self.meta = self.hdulist['primary'].header
        
    def close(self, output_verify='exception'):
        if self.is_open():
            self.hdulist['primary'].data = self.data
            self.hdulist.close(output_verify=output_verify)
        self._open = False
        self.data = None
        self.hdulist = None
        
    def __copy__(self):
        new = Image(filename=self.filename)
        return new

    def __str__(self):
        return 'Image(filename="%s")' % (self.filename)
    
    def __getstate__(self):
        return dict(data=None, meta=None, filename=self.filename,
                    _open=False, hdulist=None)
