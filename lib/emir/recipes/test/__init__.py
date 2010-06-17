
import logging
import time 
import sys
import os
import warnings
import shutil
import copy

import numpy
import pyfits

import image
from flow import SerialFlow
from node import Node
from processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from processing import compute_median, SextractorObjectMask
from numina.logger import captureWarnings
from numina.array.combine import median
from numina.array import subarray_match
from worker import para_map
from array import combine_shape, resize_array, flatcombine, correct_flatfield

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger("numina.processing")
_logger.setLevel(logging.DEBUG)
_logger = logging.getLogger("numina")
_logger.setLevel(logging.DEBUG)
    
def name_redimensioned_images(label, ext='.fits'):
    dn = 'emir_%s_base%s' % (label, ext)
    mn = 'emir_%s_mask%s' % (label, ext)
    return dn, mn
   
def segmask_naming(iteration):
    def namer(img_):
        return "emir_check%02d.fits" % iteration
        
    return namer

def name_segmask(iteration):
    return "emir_check%02d.fits" % iteration

def name_skyflat(label, iteration, ext='.fits'):
    dn = 'emir_%s_iter%02d%s' % (label, iteration, ext)
    return dn

def name_object_mask(label, iteration, ext='.fits'):
    return 'emir_%s_omask_iter%02d%s' % (label, iteration + 1, ext)

def copy_img(img, dst):
    shutil.copy(img.filename, dst)
    newimg = copy.copy(img)
    newimg.filename = dst
    return newimg

def compute_median_background(img, omask, region):
    d = img.data[region]
    m = omask.data[region]
    median_sky = numpy.median(d[m == 0])
    return median_sky

class WorkData(object):
    def __init__(self, img=None, mask=None, omask=None):
        super(WorkData, self).__init__()
        self._imgs = []
        self.dum = []
        self._masks = []
        self._omasks = []
        self.local = {}
        
        if img is not None:
            self._imgs.append(img)
        if mask is not None:
            self._masks.append(mask)
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
        newimg = copy_img(self.img, dst)
        self._imgs.append(newimg)
        return newimg
    
    def copy_mask(self, dst):
        img = self.mask
        newimg = copy_img(img, dst)
        self._masks.append(newimg)
        return newimg
    
    def copy_omask(self, dst):
        img = self.omask
        newimg = copy_img(img, dst)
        self._omasks.append(newimg)
        return newimg    

def f_flow1(od, flow):
    img = od._imgs[-1]
    img.open(mode='update')
    try:
        img = flow(img)
        return od        
    finally:
        img.close(output_verify='fix')
        
def f_resize_images(od, finalshape):
    n, d = name_redimensioned_images(od.local['label'])
    od.copy_img(n)
    od.copy_mask(d)

    shape = od.local['baseshape']
    region, _ign = subarray_match(finalshape, od.local['noffset'], shape)
    od.local['region'] = region
    # Resize image
    img = od.img
    img.open(mode='update')
    try:
        img.data = resize_array(img.data, finalshape, region)
        _logger.debug('%s resized', img)
    finally:
        _logger.debug('Closing %s', img)
        img.close()

    # Resize mask
    mask = od.mask    
    mask.open(mode='update')
    try:
        mask.data = resize_array(mask.data, finalshape, region)
        _logger.debug('%s resized', mask)
    finally:
        mask.close()
        _logger.debug('Closing %s', mask)
    return od        

def f_create_omasks(od):
    mask = od.mask
    dst = name_object_mask(od.local['label'], iteration=-1)
    shutil.copy(mask.filename, dst)
    newimg = image.Image(filename=dst)
    od._omasks.append(newimg)

def f_flow4(od):
    img = od.img
    mask = od.mask
    img.open(mode='readonly')
    mask.open(mode='readonly')
    region = od.local['region']
    try:
        d = img.data[region]
        m = mask.data[region]
        value = numpy.median(d[m == 0])
        _logger.debug('median value of %s is %f', img, value)
        od.local['median_scale'] = value
        return value        
    finally:
        img.close()
        mask.close()

def f_flow5(od, iteration, sf_data):

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
        
def f_flow6(od):
    img = od.img
    img.open()
    try:
        omask = od.omask
        omask.open()
        try:
            sky = compute_median_background(img, omask, od.local['region'])
            _logger.debug('median sky value is %f', sky)
            od.local['median_sky'] = sky
            return sky
        finally:
            omask.close()
    finally:
        img.close()
        
def f_merge_mask(od, obj_mask, iteration):
    '''Merge bad pixel _masks with object _masks'''
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

def run(pv, nthreads=4, niteration=4):

    extinction = 0
    airmass_keyword = 'AIRMASS'
    
    odl = []
    
    for i in pv['inputs']['images']:
        img = image.Image(filename=i)
        mask = image.Image(filename='bpm.fits')
        om = image.Image(filename='bpm.fits')
        
        od = WorkData(img, mask, om)
        
        label, _ext = os.path.splitext(i)
        od.local['label'] = label
        od.local['offset'] = pv['inputs']['images'][i][1]
        
        odl.append(od)
    
    offsets = [od.local['offset'] for od in odl]
    
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
    
    
    
    finalshape, offsetsp = combine_shape(image_shapes, offsets)
    
    for od, off in zip(odl, offsetsp):
        od.local['noffset'] = off
    
    # Initialize processing nodes, step 1
    dark_data = pyfits.getdata(pv['inputs']['master_dark'])    
    flat_data = pyfits.getdata(pv['inputs']['master_flat'])
    
    sss = SerialFlow([
                      DarkCorrector(dark_data),
                      NonLinearityCorrector(pv['optional']['linearity']),
                      FlatFieldCorrector(flat_data)],
                      )
    
    _logger.info('Basic processing')    
    para_map(lambda x : f_flow1(x, sss), odl, nthreads=nthreads)

    para_map(lambda x: f_resize_images(x, finalshape), odl, nthreads=nthreads)
    
    para_map(f_create_omasks, odl, nthreads=nthreads)
    
            
    for iteration in range(0, niteration):
        
        _logger.info('Iter %d, superflat correction (SF)', iteration)
        _logger.info('Iter %d, SF: computing scale factors', iteration)
        scales = para_map(f_flow4, odl, nthreads=nthreads)
        # Operation to create an intermediate sky flat
        
        try:
            map(lambda x: x.img.open(mode='readonly'), odl)
            map(lambda x: x.mask.open(mode='readonly'), odl)
            _logger.info("Iter %d, SF: combining the images without offsets", iteration)
            data = [od.img.data[od.local['region']] for od in odl]
            masks = [od.mask.data[od.local['region']] for od in odl]
            sf_data = flatcombine(data, masks, scales)
        finally:
            map(lambda x: x.img.close(), odl)
            map(lambda x: x.mask.close(), odl)
    
        # We are saving here only data part
        # FIXME, this should be done better
        pyfits.writeto('emir_sf.iter.%02d.fits' % iteration, sf_data[0], clobber=True)
    
        # Step 3, apply superflat
        _logger.info("Iter %d, SF: apply superflat", iteration)

        para_map(lambda x: f_flow5(x, iteration, sf_data), odl, nthreads=nthreads)
        
        # Compute sky backgrounf correction
        _logger.info('Iter %d, sky correction (SC)', iteration)    
        _logger.info('Iter %d, SC: computing simple sky', iteration)
        import pickle
        f = open('odl.pkl', 'wb')
        try:
            pickle.dump(odl, f)
        finally:
            f.close()
        
        
        
        
        skyback = para_map(f_flow6, odl, nthreads=nthreads)

        imgslll = [od.img for od in odl]
        mskslll = [od.mask for od in odl]
                
        try:
            map(lambda x: x.open(), imgslll)
            map(lambda x: x.open(), mskslll)
            _logger.info("Iter %d, Combining the images", iteration)
    
            # Write a node for this
            airmasses = [od.local['airmass'] for od in odl]
            extinc = [pow(10, 0.4 * am * extinction)  for am in airmasses]
            data = [i.data for i in imgslll]
            masks = [i.data for i in mskslll]
            sf_data = median(data, masks, zeros=skyback, scales=extinc, dtype='float32')
    
            # We are saving here only data part
            pyfits.writeto('emir_result.%02d.fits' % iteration, sf_data[0], clobber=True)
        finally:
            map(lambda x: x.close(), imgslll)
            map(lambda x: x.close(), mskslll)
            
        _logger.info('Iter %d, generating objects _masks', iteration)
        sex_om = SextractorObjectMask(segmask_naming(iteration))
    
        obj_mask = sex_om(sf_data[0])
    
        _logger.info('Iter %d, merging object _masks with masks', iteration)
    
        para_map(lambda x: f_merge_mask(x, obj_mask, iteration), odl, nthreads=nthreads)
        
        _logger.info('Iter %d, finished', iteration)

def run2(pv, nthreads=4, niteration=4):
    import pickle

    f = open('odl.pkl', 'rb')
    try:
        odl = pickle.load(f)
    finally:
        f.close()
        
    odl.sort(key=lambda odl: odl.local['label'])
    nox = 5

    def print_related(od):
        bw, fw = od.local['sky_related']
        print od.local['label'],':',
        for b in bw:
                print b.local['label'],
        print '|',
        for f in fw:
                print f.local['label'],
        print

    
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
        #print_related(od)
    
    # _logger.info('Opening sky files')
    try:
        map(lambda x: x.img.open(mode='readonly', memmap=True), odl)
        map(lambda x: x.mask.open(mode='readonly', memmap=True), odl)
        for od in odl:
            bw, fw = od.local['sky_related']
            print_related(od)
    finally:
        map(lambda x: x.img.close(), odl)
        map(lambda x: x.mask.close(), odl)


if __name__ == '__main__':
    import os.path

    captureWarnings(True)
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    
    pv = {'inputs' :  { 'images':  
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
                        'master_dark': 'Dark50.fits',
                        'master_flat': 'flat.fits',
                        'master_bpm': 'bpm.fits'
                        },
          'optional' : {'linearity': [1.00, 0.00],
                        'extinction,': 0.05,
                        }          
    }
    
    # Changing the offsets
    # x, y -> -y, -x
    for k in pv['inputs']['images']:
        m_, o_, s_ = pv['inputs']['images'][k]
        x, y = o_
        o_ = -y, -x
        pv['inputs']['images'][k] = (m_, o_, s_)

    run2(pv, nthreads=1, niteration=2)
