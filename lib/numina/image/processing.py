#
# Copyright 2008-2010 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 


import logging
import time

import pyfits
from numpy import polyval



# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("numina.processing")

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

def version_filename(file_spec):
    import os
    if os.path.isfile(file_spec):
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
                return new_file
        raise RuntimeError("Can't backup %r, all names taken" % (file_spec,))
    return ""

class Corrector:
    def __init__(self, flag, comment, dtype='float32'):
        self.flag = flag
        self.comment = comment
        self.name = ''
        self.dtype = dtype 

    def mark_as_processed(self, hdulist):
        hdulist[0].header.update(self.flag, time.asctime(), self.comment)

    def check_if_processed(self, hdulist):
        primary = hdulist['PRIMARY']
        # check if is already processed
        if not primary.header.has_key(self.flag):
            return False
        return True
    
    def correct(self, hdulist):
        raise NotImplementedError

class BiasCorrector(Corrector):
    def __init__(self, biasdata, dtype='float32'):
        super(BiasCorrector, self).__init__('NUM-BS', 'Bias removed with numina', dtype)
        self.biasdata = biasdata

    def correct(self, hdulist):
        primary = hdulist['PRIMARY']
        primary.data -= self.biasdata
        primary.data = primary.data.astype(self.dtype)
        return True


class DarkCorrector(Corrector):
    def __init__(self, darkdata, dtype='float32'):
        super(DarkCorrector, self).__init__('NUM-DK', 'Dark removed with numina', dtype)
        self.darkdata = darkdata

    def correct(self, hdulist):
        primary = hdulist['PRIMARY']
        primary.data -= self.darkdata
        primary.data = primary.data.astype(self.dtype)
        return True

class NonLinearityCorrector(Corrector):
    def __init__(self, polynomial, dtype='float32'):
        super(NonLinearityCorrector, self).__init__('NUM-LIN', 'Non-linearity removed with numina', dtype)
        self.polynomial = polynomial
                
    def correct(self, hdulist):
        primary = hdulist['PRIMARY']
        primary.data = polyval(self.polynomial, primary.data)
        primary.data = primary.data.astype(self.dtype)
        return True
        
        
class FlatFieldCorrector(Corrector):
    def __init__(self, flatdata, dtype='float32'):
        super(FlatFieldCorrector, self).__init__('NUM-FF', 'Flat field removed with numina', dtype)
        self.flatdata = flatdata

    def correct(self, hdulist):
        primary = hdulist['PRIMARY']
        primary.data /= self.flatdata
        primary.data = primary.data.astype(self.dtype)
        return True
            
def generic_processing(inputs, correctors, backup=False, output_verify='ignore', outputs=None):
    
    if outputs is None:
        outputs = inputs
    
    for input, output in zip(inputs, outputs):
        do_backup = True
        # Assuming that input is a filename...
        hdulist = pyfits.open(input)
        try:
            for corrector in correctors:
                if not corrector.check_if_processed(hdulist):
                    _logger.info("Processing %s with %s", input, corrector)
                    
                    if corrector.correct(hdulist):
                        if backup and do_backup:
                            # I'm doing a backup, so I 
                            _logger.debug("Doing the backup")
                            version_file(input)
                            do_backup = False
                            
                        corrector.mark_as_processed(hdulist)
                
                    hdulist.writeto(output, clobber=True, output_verify=output_verify)
                else:
                    _logger.info("Image %s already processed by %s, skipping", input, corrector)
        finally:
            hdulist.close()
