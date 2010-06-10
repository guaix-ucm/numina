
import logging
import time 
import sys
import os
import warnings

import numpy

import worker
import image
from node import AdaptorNode, IdNode, OutputSelector
from flow import SerialFlow, ParallelFlow
from node import Node
from processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from processing import OpenNode, CloseNode, BackupNode, SaveAsNode, ResizeNode
from processing import compute_median
from utils import iterqueue
from numina.logger import captureWarnings

logging.basicConfig(level=logging.ERROR)

# Ignore pyfits Overwrite warning, pyfits >= 2.3 only
#warnings.filterwarnings('ignore', '.*overwrite.*',)

captureWarnings(True)

_logger = logging.getLogger("numina.processing")
_logger.setLevel(logging.DEBUG)
_logger = logging.getLogger("numina")
_logger.setLevel(logging.DEBUG)

    
def basic_naming(img):
    f = img.datafile
    base, ext = os.path.splitext(f)
    d = 'emir_%s_base%s' % (base, ext)
    m = 'emir_%s_mask%s' % (base, ext)
    return (d, m)
   
def segmask_naming(iteration):
    def namer(img):
        return "emir_check%02d.fits" % iteration
        
    return namer

def skyflat_naming(iteration):
    def namer(img):
        f = img.datafile
        base, ext = os.path.splitext(f)
        if base.startswith('emir_'):
            d = '%s_iter%02d%s' % (base, iteration, ext)
        else:
            d = 'emir_%s_iter%02d%s' % (base, iteration, ext)
        m = img.maskfile
        return d, m
    
    return namer

def omask_naming(iteration):
    def namer(img):
        f = img.maskfile
        d = '%s.omask' % f
        return d
    
    return namer


class SextractorObjectMask(Node):
    def __init__(self, namegen):
        self.namegen = namegen
        
    def _run(self, array):
        import tempfile
        import subprocess
        import os.path
    
        checkimage = self.namegen(None)
    
        # A temporary file used to store the array in fits format
        tf = tempfile.NamedTemporaryFile(prefix='emir_', dir='.')
        pyfits.writeto(filename=tf, data=array)
        
        # Run sextractor, it will create a image called check.fits
        # With the segmentation mask inside
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

    
class MaskMerging(Node):
    def __init__(self, objmask, namegen):
        self.objmask = objmask
        self.namegen = namegen
        
    def _run(self, img):
        newdata = (img.mask != 0) | (self.objmask != 0)
        newdata = newdata.astype('int')
        newfile = self.namegen(img)
        pyfits.writeto(newfile, newdata, output_verify='silentfix', clobber=True)
        _logger.info('Mask %s merged with object mask into %s', img.maskfile, newfile)
        return img

if __name__ == '__main__':
    import pyfits
    import os.path
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    
    def combine_shape(shapes, offsets):
        # Computing final image size and new offsets
        sharr = numpy.asarray(shapes)
    
        offarr = numpy.asarray(offsets)        
        ucorners = offarr + sharr
        ref = offarr.min(axis=0)     
        finalshape = ucorners.max(axis=0) - ref 
        offsetsp = offarr - ref
        return (finalshape, offsetsp)
    

    

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
        m, o, s = pv['inputs']['images'][k]
        x, y = o
        o = -y, -x
        pv['inputs']['images'][k] = (m, o, s)

# -----------------------------------------
 
    _logger = logging.getLogger('numina')
    _logger.setLevel(logging.DEBUG)

    nthreads = 1

    imagesl = []
    imagesd = {}
    procl = []
    procd = {}
    
    # Reading data:
    initd = []
    
    for i in pv['inputs']['images']:
        initd.append(image.EmirImage(datafile=i,
                                     maskfile=pv['inputs']['images'][i][0],
                                     offset=pv['inputs']['images'][i][1]))
    
    
    offsets = [im.offset for im in initd]
    finalshape, offsetsp = combine_shape((2048, 2048), offsets)
    
    for im, off in zip(initd, offsetsp):
        im.noffset = off
    
    imagesl.append(initd)
    
    # Initialize processing nodes, step 1
    dark_data = pyfits.getdata(pv['inputs']['master_dark'])    
    flat_data = pyfits.getdata(pv['inputs']['master_flat'])

    # basic corrections in the first pass
    nodes = SerialFlow([OpenNode(mode='update'),
                         DarkCorrector(dark_data),
                         NonLinearityCorrector(pv['optional']['linearity']),
                         FlatFieldCorrector(flat_data),
                         #BackupNode(),
                         CloseNode(output_verify='fix')])
    
    procl.append(nodes)
    procd['basic'] = nodes
    
    _logger.info('Basic processing')
    result = worker.para_map(procd['basic'], imagesl[0], nthreads=nthreads)
    
    imagesl.append(result)
    imagesd['basic'] = result
    
    
    del dark_data
    del flat_data
    
    nodes = SerialFlow([OpenNode(), SaveAsNode(namegen=basic_naming),
                         ParallelFlow([CloseNode(),
                                          SerialFlow([OpenNode(mode='update'),
                                                      ResizeNode(finalshape),
                                                      CloseNode(output_verify='fix')])])])
    
    procl.append(nodes)
    procd['resize'] = nodes
        
    result = worker.para_map(procd['resize'], imagesl[1], nthreads=nthreads)
    # store2 is a list of tuples
    # we need the second component
    ignore_, result = zip(*result)
    imagesl.append(result)
        
    iteration = 0
    
    _logger.info('Iter %d, superflat correction (SF)', iteration)

    # Step 2, compute superflat
    _logger.info('Iter %d, SF: computing scale factors', iteration)
    # Actions:
    nodes = SerialFlow(# -> Runs its internal nodes one after another
                        [OpenNode(mode='readonly'), # -> Load the data
                         AdaptorNode(compute_median), # -> Call the compute_median on data
                                                      # -> It returns the median AND
                                                      # -> the data
                         ParallelFlow([ # A series of nodes in parallel
                                          IdNode(), # gets the median and returns the median
                                          CloseNode() # gets the data and closes the file in the end
                                          ] 
                                         )
                         ]
    )
    procd['scale'] = nodes
    
    result = worker.para_map(procd['scale'], imagesl[2], nthreads=nthreads)
    
    # We get the scale factor 
    scales, ignore_ = zip(*result)
    # Operation to create an intermediate sky flat
        
    result = map(OpenNode(), imagesl[2])
    _logger.info("Iter %d, SF: combining the images without offsets", iteration)
    
    # Write a node for this
    sf_data = image.combine('median', result, scales=scales)
    # Zero masks
    mm = sf_data[0] == 0
    sf_data[0][mm] = 1
    
    # We are saving here only data part
    pyfits.writeto('emir_sf.iter.%02d.fits' % iteration, sf_data[0], clobber=True)
    
    map(CloseNode(), result)

    #sys.exit(0)
    
    # Step 3, apply superflat
    _logger.info("Iter %d, SF: apply superflat", iteration)

    
    nodes = SerialFlow([OpenNode(),
                        SaveAsNode(namegen=skyflat_naming(iteration)),
                        ParallelFlow([CloseNode(),
                                         SerialFlow([OpenNode(mode='update'),
                                                     FlatFieldCorrector(sf_data[0],
                                                                        mark=False,
                                                                        region=True),
                                                      CloseNode()])])])

    procd['sfproc'] = nodes
    result = worker.para_map(procd['sfproc'], imagesl[2], nthreads=nthreads)
    ignore_, skys = zip(*result)
    imagesl.append(skys)
    
    _logger.info('Iter %d, sky correction (SC)', iteration)    
    
    def compute_sky_simple(img):
        skyval = img.data[img.region]
        skymask = img.mask[img.region] != 0
        median_sky = numpy.median(skyval[skymask])
        return median_sky, img
    
    _logger.info('Iter %d, SC: computing simple sky', iteration)
    
    # Compute simple sky
    # Actions:
    nodes = SerialFlow(# -> Runs its internal nodes one after another
                        [OpenNode(mode='readonly'), # -> Load the data
                         AdaptorNode(compute_sky_simple),
                                                      # -> It returns the median AND
                                                      # -> the data
                         ParallelFlow([ # A series of nodes in parallel
                                          IdNode(), # gets the median and returns the median
                                          CloseNode() # gets the data and closes the file in the end
                                          ] 
                                        ),
                                        OutputSelector((0,))
                         ]
    )
    
    procd['skysimple'] = nodes
    
    skyback = worker.para_map(procd['skysimple'], imagesl[3], nthreads=nthreads)
    
    result = map(OpenNode(), imagesl[3])
    _logger.info("Iter %d, Combining the images", iteration)
    
    # Write a node for this
    extinc = [pow(10, 0.4 * 1.2 * 0.01)  for i in result]
    sf_data = image.combine2('median', result, zeros=skyback, scales=extinc, dtype='float32')
    
    # We are saving here only data part
    pyfits.writeto('result.%02d.fits' % iteration, sf_data[0], clobber=True)
    
    map(CloseNode(), result)
    
    _logger.info('Iter %d, generating objects mask', iteration)
    sex_om = SextractorObjectMask(segmask_naming(iteration))
    
    
    print sf_data[0].shape
    print sf_data[0]
    
    obj_mask = sex_om(sf_data[0])
    
    _logger.info('Iter %d, merging object mask with masks', iteration)
    #map(lambda f: mask_merging(f, obj_mask, itern), self.images)
    # We need to store this result
    dflow = SerialFlow([OpenNode(mode='readonly'),
                        MaskMerging(obj_mask, omask_naming(iteration)),
                        CloseNode()
                        ])
    
    procd['mmerge'] = dflow
    
    worker.para_map(procd['mmerge'], imagesl[3], nthreads=nthreads)
    
    _logger.info('Iter %d, finished', iteration)
    
    sys.exit(0)
