
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

'''Recipe for the reduction of imaging mode observations.

Recipe to reduce observations obtained in imaging mode, considering different
possibilities depending on the size of the offsets between individual images.
In particular, the following strategies are considered: stare imaging, nodded
beamswitched imaging, and dithered imaging. 

A critical piece of information here is a table that clearly specifies which
images can be labeled as *science*, and which ones as *sky*. Note that some
images are used both as *science* and *sky* (when the size of the targets are
small compared to the offsets).

**Observing modes:**
 
 * StareImage
 * Nodded/Beam-switched images
 * Dithered images 


**Inputs:**

 * Science frames + [Sky Frames]
 * An indication of the observing strategy: **stare image**, **nodded
   beamswitched image**, or **dithered imaging**
 * A table relating each science image with its sky image(s) (TBD if it's in 
   the FITS header and/or in other format)
 * Offsets between them (Offsets must be integer)
 * Master Dark 
 * Bad pixel mask (BPM) 
 * Non-linearity correction polynomials 
 * Master flat (twilight/dome flats)
 * Master background (thermal background, only in K band)
 * Exposure Time (must be the same in all the frames)
 * Airmass for each frame
 * Detector model (gain, RN, lecture mode)
 * Average extinction in the filter
 * Astrometric calibration (TBD)

**Outputs:**

 * Image with three extensions: final image scaled to the individual exposure
   time, variance  and exposure time map OR number of images combined (TBD)

**Procedure:**

Images are corrected from dark, non-linearity and flat. Then, an iterative
process starts:

 * Sky is computed from each frame, using the list of sky images of each
   science frame. The objects are avoided using a mask (from the second
   iteration on).

 * The relative offsets are the nominal from the telescope. From the second
   iteration on, we refine them using objects of appropriate brightness (not
   too bright, not to faint).

 * We combine the sky-subtracted images, output is: a new image, a variance
   image and a exposure map/number of images used map.

 * An object mask is generated.

 * We recompute the sky map, using the object mask as an additional input. From
   here we iterate (typically 4 times).

 * Finally, the images are corrected from atmospheric extinction and flux
   calibrated.

 * A preliminary astrometric calibration can always be used (using the central
   coordinates of the pointing and the plate scale in the detector). A better
   calibration might be computed using available stars (TBD).

'''

import logging
import time 
import sys
import os
import warnings
import shutil
import copy

import numpy
import pyfits

import registry
import image
from flow import SerialFlow
from processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
#from processing import compute_median
from numina.logger import captureWarnings
from numina.array.combine import median
from numina.array import subarray_match
from worker import para_map
from array import combine_shape, resize_array, flatcombine, correct_flatfield
from array import compute_median_background, compute_sky_advanced, create_object_mask
import numina.recipes as nr
#from numina.exceptions import RecipeError
#from numina.image.processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from numina.image.processing import generic_processing
#from numina.array.combine import median
from emir.instrument.headers import EmirImageCreator
import numina.qa as QA
from numina.exceptions import RecipeError

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger("numina.processing")
_logger.setLevel(logging.DEBUG)
_logger = logging.getLogger("numina")
_logger.setLevel(logging.DEBUG)


_logger = logging.getLogger("emir.recipes")

class WorkData(object):
    '''The data processed during the run of the recipe.
    
    Each instance contains the science image, its mask
    and the object mask produced during the processing.
    
    The local member contains data local to each particular image
     '''
    def __init__(self, img=None, mask=None, omask=None):
        super(WorkData, self).__init__()
        self._imgs = []
        self._masks = []
        self._omasks = []
        self.local = {}
        
        if img is not None:
            self._imgs.append(img)
            self.base = img
        if mask is not None:
            self._masks.append(mask)
            self.mase = mask
        if omask is not None:
            self._omasks.append(omask)

    @property
    def img(self):
        return self._imgs[-1]
    
    @property
    def mask(self):
        return self._masks[-1]
    
    @property
    def omask(self):
        return self._omasks[-1]    
    
    def copy_img(self, dst):
        newimg = self.img.copy(dst)
        self._imgs.append(newimg)
        return newimg
    
    def copy_mask(self, dst):
        img = self.mask
        newimg = img.copy(dst)
        self._masks.append(newimg)
        return newimg
    
    def copy_omask(self, dst):
        img = self.omask
        newimg = img.copy(dst)
        self._omasks.append(newimg)
        return newimg
    
    def append_img(self, newimg):
        self._imgs.append(newimg)
        
    def append_mask(self, newimg):
        self._masks.append(newimg)
        
    def append_omask(self, newimg):
        self._omasks.append(newimg)        

def print_related(od):
    bw, fw = od.local['sky_related']
    print od.local['label'],':',
    for b in bw:
        print b.local['label'],
    print '|',
    for f in fw:
        print f.local['label'],
    print

def related(odl, nimages=5):
    
    odl.sort(key=lambda odl: odl.local['label'])
    nox = nimages

    for idx, od in enumerate(odl[0:nox]):
        bw = odl[0:2 * nox + 1][0:idx]
        fw = odl[0:2 * nox + 1][idx + 1:]
        od.local['sky_related'] = (bw, fw)
        #print_related(od)

    for idx, od in enumerate(odl[nox:-nox]):
        bw = odl[idx:idx + nox]
        fw = odl[idx + nox + 1:idx + 2 * nox + 1]
        od.local['sky_related'] = (bw, fw)
        #print_related(od)
 
    for idx, od in enumerate(odl[-nox:]):
        bw = odl[-2 * nox - 1:][0:idx + nox + 1]
        fw = odl[-2 * nox - 1:][nox + idx + 2:]
        od.local['sky_related'] = (bw, fw)
    
    return odl

# Different intermediate images are created during the run of the recipe
# These functions are used to give them a meaningful name
# If a name with meaning is not required, the function name_uuidname
# can be used

def name_redimensioned_images(label, iteration, ext='.fits'):
    dn = 'emir_%s_base_i%02d%s' % (label, iteration, ext)
    mn = 'emir_%s_mask_i%02d%s' % (label, iteration, ext)
    return dn, mn

def name_segmask(iteration):
    return "emir_check_i%02d.fits" % iteration

def name_skyflat(label, iteration, ext='.fits'):
    dn = 'emir_%s_f_i%02d%s' % (label, iteration, ext)
    return dn

def name_skysub(label, iteration, ext='.fits'):
    dn = 'emir_%s_fs_i%02d%s' % (label, iteration, ext)
    return dn

def name_object_mask(label, iteration, ext='.fits'):
    return 'emir_%s_omask_i%02d%s' % (label, iteration + 1, ext)

def name_uuidname():
    import uuid
    return uuid.uuid4().hex

# Flows, these functions are run inside para_map, so
# they can be run in parallel

def f_basic_processing(od, flow):
    img = od.img
    img.open(mode='update')
    try:
        img = flow(img)
        return od        
    finally:
        img.close(output_verify='fix')
        
def f_resize_images(od, iteration, finalshape):
    n, d = name_redimensioned_images(od.local['label'], iteration)
    
    newimg = od.base.copy(n)
    newmsk = od.mase.copy(d)
    
    shape = od.local['baseshape']
    region, _ign = subarray_match(finalshape, od.local['noffset'], shape)
    
    if od.local.has_key('region'):
        assert od.local['region'] == region
        
    od.local['region'] = region
    # Resize image

    newimg.open(mode='update')
    try:
        newimg.data = resize_array(newimg.data, finalshape, region)
        _logger.debug('%s resized to %s', newimg, finalshape)
    finally:
        newimg.close()
    
    newmsk.open(mode='update')
    try:
        newmsk.data = resize_array(newmsk.data, finalshape, region)
        _logger.debug('%s resized to %s', newmsk, finalshape)
    finally:
        newmsk.close()
        
    od.append_img(newimg)
    od.append_mask(newmsk)
    
    return od        

def f_create_omasks(od):
    dst = name_object_mask(od.local['label'], iteration=-1)
    newimg = od.mask.copy(dst)
    od.append_omask(newimg)

def f_flow4(od):
    region = od.local['region']
    
    od.img.open(mode='readonly')
    try:
        od.mask.open(mode='readonly')        
        try:
            d = od.img.data[region]
            m = od.mask.data[region]
            value = numpy.median(d[m == 0])
            _logger.debug('median value of %s is %f', od.img, value)
            od.local['median_scale'] = value
            return od        
        finally:
            od.mask.close()
    finally:
        od.img.close()
        

def f_sky_flatfielding(od, iteration, sf_data):

    dst = name_skyflat(od.local['label'], iteration)
    od.copy_img(dst)
    img = od.img
    region = od.local['region']
    img.open(mode='update')
    try:
        img.data[region] = correct_flatfield(img.data[region], sf_data[0])
    finally:
        img.close()
    
    return od
        
def f_sky_removal_simple(od, iteration):
    dst = name_skysub(od.local['label'], iteration)
    od.copy_img(dst)
    #
    od.img.open(mode='update')
    try:
        od.omask.open()
        try:
            sky = compute_median_background(od.img, od.omask, od.local['region'])
            _logger.debug('median sky value is %f', sky)
            od.local['median_sky'] = sky
            
            _logger.info('Iter %d, SC: subtracting sky', iteration)
            region = od.local['region']
            od.img.data[region] -= od.local['median_sky']
            
            return od
        finally:
            od.omask.close()
    finally:
        od.img.close()
        
def f_sky_removal_advanced(od, iteration):
    dst = name_skysub(od.local['label'], iteration)
    result = od.img.copy(dst)
    #
    result.open(mode='update')
    try:
        bw, fw = od.local['sky_related']
        rep = bw + fw
        data = [i.img.data[i.local['region']] for i in rep]
        omasks = [i.omask.data[i.local['region']] for i in rep]
        sky_b = compute_sky_advanced(data, omasks)
        
        _logger.info('Iter %d, SC: saving sky', iteration)
        filename = 'emir_%s_skyb_i%02d.fits' % (od.local['label'], iteration)
        pyfits.writeto(filename, sky_b, clobber=True)
        
        _logger.info('Iter %d, SC: subtracting sky', iteration)
        region = od.local['region']
        result.data[region] -= sky_b
        
        # We can't modify the current od.img until all
        # the images are processed
        od.local['skysub'] = result
        
        return od
    finally:
        result.close()

def f_merge_mask(od, obj_mask, iteration):
    '''Merge bad pixel masks with object masks'''
    dst = name_object_mask(od.local['label'], iteration)
    od.copy_omask(dst)
    #
    img = od.omask
    img.open(mode='update')
    try:
        img.data = (img.data != 0) | (obj_mask != 0)
        img.data = img.data.astype('int')
        return od
    finally:
        img.close()
                

class Recipe(object):
    param = ['master_dark',
        'master_bpm',
        'master_dark',
        'master_flat',
        'nonlinearity',
        'extinction',
        'nthreads',
        'images',
    ]
    def __init__(self):
        self.values = {}
        for p in self.param:
            self.values[p] = registry.lookup('1', p)

    def process(self):
        extinction = self.values['extinction']
        nthreads = self.values['nthreads']
        niteration = 4
        airmass_keyword = 'AIRMASS'
        
        odl = []
        
        for i in self.values['images']:
            img = image.Image(filename=i)
            mask = image.Image(filename='bpm.fits')
            om = image.Image(filename='bpm.fits')
            
            od = WorkData(img, mask, om)
            
            label, _ext = os.path.splitext(i)
            od.local['label'] = label
            od.local['offset'] = self.values['images'][i][1]
            
            odl.append(od)
                
        image_shapes = []
        
        for od in odl:
            img = od.img
            img.open(mode='readonly')
            try:
                _logger.debug("opening image %s", img.filename)
                image_shapes.append(img.data.shape)
                od.local['baseshape'] = img.data.shape
                _logger.debug("shape of image %s is %s", img.filename, image_shapes[-1])
                od.local['airmass'] = img.meta[airmass_keyword]
                _logger.debug("airmass of image %s is %f", img.filename, od.local['airmass'])
            finally:
                _logger.debug("closing image %s", img.filename)
                img.close()
            
        # Initialize processing nodes, step 1
        dark_data = pyfits.getdata(self.values['master_dark'])    
        flat_data = pyfits.getdata(self.values['master_flat'])
        
        sss = SerialFlow([
                          DarkCorrector(dark_data),
                          NonLinearityCorrector(pv['optional']['linearity']),
                          FlatFieldCorrector(flat_data)],
                          )
        
        _logger.info('Basic processing')    
        para_map(lambda x : f_basic_processing(x, sss), odl, nthreads=nthreads)
        
        sf_data = None
        
        for iteration in range(1, niteration + 1):
            
            _logger.info('Iter %d, computing offsets', iteration)
            
            offsets = [od.local['offset'] for od in odl]
            finalshape, offsetsp = combine_shape(image_shapes, offsets)
        
            for od, off in zip(odl, offsetsp):
                od.local['noffset'] = off
            
            _logger.info('Iter %d, resizing images and mask', iteration)
            para_map(lambda x: f_resize_images(x, iteration, finalshape), odl, nthreads=nthreads)
        
            _logger.info('Iter %d, initialize object masks', iteration)
            para_map(f_create_omasks, odl, nthreads=nthreads)
            
            if sf_data is not None:
                _logger.info('Iter %d, generating objects masks', iteration)    
                obj_mask = create_object_mask(sf_data[0], name_segmask(iteration))
                
                _logger.info('Iter %d, merging object masks with masks', iteration)
                para_map(lambda x: f_merge_mask(x, obj_mask, iteration), odl, nthreads=nthreads)
            
            _logger.info('Iter %d, superflat correction (SF)', iteration)
            _logger.info('Iter %d, SF: computing scale factors', iteration)
            
            para_map(f_flow4, odl, nthreads=nthreads)
            # Operation to create an intermediate sky flat
            
            try:
                map(lambda x: x.img.open(mode='readonly'), odl)
                map(lambda x: x.mask.open(mode='readonly'), odl)
                _logger.info("Iter %d, SF: combining the images without offsets", iteration)
                data = [od.img.data[od.local['region']] for od in odl]
                masks = [od.mask.data[od.local['region']] for od in odl]
                scales = [od.local['median_scale'] for od in odl]
                sf_data = flatcombine(data, masks, scales)
            finally:
                map(lambda x: x.img.close(), odl)
                map(lambda x: x.mask.close(), odl)
        
            # We are saving here only data part
            # FIXME, this should be done better
            pyfits.writeto('emir_sf_i%02d.fits' % iteration, sf_data[0], clobber=True)
        
            # Step 3, apply superflat
            _logger.info("Iter %d, SF: apply superflat", iteration)
    
            para_map(lambda x: f_sky_flatfielding(x, iteration, sf_data), odl, nthreads=nthreads)
            
            # Compute sky backgrounf correction
            _logger.info('Iter %d, sky correction (SC)', iteration)
            
            if iteration in [1]:
                _logger.info('Iter %d, SC: computing simple sky', iteration)
            
                para_map(lambda x: f_sky_removal_simple(x, iteration), odl, nthreads=nthreads)
                                
            else:
                _logger.info('Iter %d, SC: computing advanced sky', iteration)
                
                nimages = 5
                
                _logger.info('Iter %d, SC: relating images with their sky backgrounds', iteration)
                odl = related(odl, nimages)
                            
                try:
                    map(lambda x: x.img.open(mode='readonly', memmap=True), odl)
                    map(lambda x: x.omask.open(mode='readonly', memmap=True), odl)
                    for od in odl:
                        f_sky_removal_advanced(od, iteration)
                finally:
                    map(lambda x: x.img.close(), odl)
                    map(lambda x: x.omask.close(), odl)
                    
                # We update img list now
                # after the sky background is computed for every image
                for od in odl:
                    od.append_img(od.local['skysub'])    
                
            imgslll = [od.img for od in odl]
            mskslll = [od.mask for od in odl]
                    
            try:
                map(lambda x: x.open(mode='readonly', memmap=True), imgslll)
                map(lambda x: x.open(mode='readonly', memmap=True), mskslll)
                _logger.info("Iter %d, Combining the images", iteration)
    
                extinc = [pow(10, 0.4 * od.local['airmass'] * extinction) for od in odl]
                data = [i.data for i in imgslll]
                masks = [i.data for i in mskslll]
                sf_data = median(data, masks, scales=extinc, dtype='float32')
    
                # We are saving here only data part
                pyfits.writeto('emir_result_i%02d.fits' % iteration, sf_data[0], clobber=True)
            finally:
                map(lambda x: x.close(), imgslll)
                map(lambda x: x.close(), mskslll)
    
                
    
        
            _logger.info('Iter %d, finished', iteration)

if __name__ == '__main__':
    import os.path
    from numina.recipes import Parameters
    import simplejson as json
    from numina.jsonserializer import to_json
#    import registry
    
    captureWarnings(True)
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    
    pv = {'inputs' :  {},
          'optional' : {'linearity': [1.00, 0.00],
                        'extinction': 0.05,
                        'niteration': 2, 
                        'master_dark': 'Dark50.fits',
                        'master_flat': 'flat.fits',
                        'master_bpm': 'bpm.fits',
                        'images':  
                       {'apr21_0046.fits': ('bpm.fits', (0, 0), ['apr21_0046.fits']),
                        'apr21_0047.fits': ('bpm.fits', (0, 0), ['apr21_0047.fits']),
                        'apr21_0048.fits': ('bpm.fits', (0, 0), ['apr21_0048.fits']),
                        'apr21_0049.fits': ('bpm.fits', (21, -23), ['apr21_0049.fits']),
                        'apr21_0051.fits': ('bpm.fits', (21, -23), ['apr21_0051.fits']),
                        'apr21_0052.fits': ('bpm.fits', (-15, -35), ['apr21_0052.fits']),
                        'apr21_0053.fits': ('bpm.fits', (-15, -35), ['apr21_0053.fits']),
                        'apr21_0054.fits': ('bpm.fits', (-15, -35), ['apr21_0054.fits']),
                        'apr21_0055.fits': ('bpm.fits', (24, 12), ['apr21_0055.fits']),
                        'apr21_0056.fits': ('bpm.fits', (24, 12), ['apr21_0056.fits']),
                        'apr21_0057.fits': ('bpm.fits', (24, 12), ['apr21_0057.fits']),
                        'apr21_0058.fits': ('bpm.fits', (-27, 18), ['apr21_0058.fits']),
                        'apr21_0059.fits': ('bpm.fits', (-27, 18), ['apr21_0059.fits']),
                        'apr21_0060.fits': ('bpm.fits', (-27, 18), ['apr21_0060.fits']),
                        'apr21_0061.fits': ('bpm.fits', (-38, -16), ['apr21_0061.fits']),
                        'apr21_0062.fits': ('bpm.fits', (-38, -16), ['apr21_0062.fits']),
                        'apr21_0063.fits': ('bpm.fits', (-38, -17), ['apr21_0063.fits']),
                        'apr21_0064.fits': ('bpm.fits', (5, 27), ['apr21_0064.fits']),
                        'apr21_0065.fits': ('bpm.fits', (5, 27), ['apr21_0065.fits']),
                        'apr21_0066.fits': ('bpm.fits', (5, 27), ['apr21_0066.fits']),
                        'apr21_0067.fits': ('bpm.fits', (32, -13), ['apr21_0067.fits']),
                        'apr21_0068.fits': ('bpm.fits', (33, -13), ['apr21_0068.fits']),
                        'apr21_0069.fits': ('bpm.fits', (32, -13), ['apr21_0069.fits']),
                        'apr21_0070.fits': ('bpm.fits', (-52, 7), ['apr21_0070.fits']),
                        'apr21_0071.fits': ('bpm.fits', (-52, 8), ['apr21_0071.fits']),
                        'apr21_0072.fits': ('bpm.fits', (-52, 8), ['apr21_0072.fits']),
                        'apr21_0073.fits': ('bpm.fits', (-3, -49), ['apr21_0073.fits']),
                        'apr21_0074.fits': ('bpm.fits', (-3, -49), ['apr21_0074.fits']),
                        'apr21_0075.fits': ('bpm.fits', (-3, -49), ['apr21_0075.fits']),
                        'apr21_0076.fits': ('bpm.fits', (-49, -33), ['apr21_0076.fits']),
                        'apr21_0077.fits': ('bpm.fits', (-49, -32), ['apr21_0077.fits']),
                        'apr21_0078.fits': ('bpm.fits', (-49, -32), ['apr21_0078.fits']),
                        'apr21_0079.fits': ('bpm.fits', (-15, 36), ['apr21_0079.fits']),
                        'apr21_0080.fits': ('bpm.fits', (-16, 36), ['apr21_0080.fits']),
                        'apr21_0081.fits': ('bpm.fits', (-16, 36), ['apr21_0081.fits'])
                        },
                        }          
    }
    
    # Changing the offsets
    # x, y -> -y, -x
    for k in pv['optional']['images']:
        m_, o_, s_ = pv['optional']['images'][k]
        x, y = o_
        o_ = -y, -x
        pv['optional']['images'][k] = (m_, o_, s_)

    p = Parameters(**pv)
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    f = open('config-d.json', 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
        
        
    repos = registry.get_repo_list()
    
    filerepo = registry.JSON_Repo('/home/spr/Datos/emir/apr21/config-d.json')

    newrepo = [filerepo] + repos

    registry.set_repo_list(newrepo)

    #run(pv, nthreads=1, niteration=4)
    
    r = Recipe()
    r.process()
    
    
