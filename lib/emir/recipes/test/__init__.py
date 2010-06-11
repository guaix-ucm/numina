
import logging
import time 
import sys
import os
import warnings

import numpy

import image
from node import AdaptorNode, IdNode, OutputSelector
from flow import SerialFlow, ParallelFlow
from node import Node
from processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from processing import OpenImage, CloseImage, BackupNode, SaveAsNode, ResizeNode
from processing import compute_median, SextractorObjectMask, CopyMask
from numina.logger import captureWarnings
from numina.array.combine import median

logging.basicConfig(level=logging.ERROR)

captureWarnings(True)

_logger = logging.getLogger("numina.processing")
_logger.setLevel(logging.DEBUG)
_logger = logging.getLogger("numina")
_logger.setLevel(logging.DEBUG)

    
def basic_naming(img):
    f = img.datafile
    _base, ext = os.path.splitext(f)
    dn = 'emir_%s_base%s' % (img.label, ext)
    mn = 'emir_%s_mask%s' % (img.label, ext)
    return dn, mn
   
def segmask_naming(iteration):
    def namer(img):
        return "emir_check%02d.fits" % iteration
        
    return namer

def skyflat_naming(iteration):
    def namer(img):
        f = img.datafile
        _base, ext = os.path.splitext(f)
        dn = 'emir_%s_iter%02d%s' % (img.label, iteration, ext)
        mn = img.maskfile
        return dn, mn
    
    return namer

def omask_naming(iteration):
    def namer(img):     
        f = img.datafile
        _base, ext = os.path.splitext(f)   
        dn = 'emir_%s_omask_iter%02d%s' % (img.label, iteration, ext)
        return dn
    
    return namer
    
class MaskMerging(Node):
    def __init__(self, objmask, namegen):
        super(MaskMerging, self).__init__()
        self.objmask = objmask
        self.namegen = namegen
        
    def _run(self, img):
        newdata = (img.mask != 0) | (self.objmask != 0)
        newdata = newdata.astype('int')
        newfile = self.namegen(img)
        pyfits.writeto(newfile, newdata, output_verify='silentfix', clobber=True)
        _logger.info('Mask %s merged with object mask into %s', img.maskfile, newfile)
        return img



def combine_shape(shapes, offsets):
    # Computing final image size and new offsets
    sharr = numpy.asarray(shapes)

    offarr = numpy.asarray(offsets)        
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)     
    finalshape = ucorners.max(axis=0) - ref 
    offsetsp = offarr - ref
    return (finalshape, offsetsp)

def compute_sky_simple(img):
    d = img.data[img.region]
    m = img.mask[img.region]
    median_sky = numpy.median(d[m == 0])
    _logger.debug('median sky value is %f', median_sky)
    return median_sky, img


if __name__ == '__main__':
    import pyfits
    import os.path
    
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
        m, o, s = pv['inputs']['images'][k]
        x, y = o
        o = -y, -x
        pv['inputs']['images'][k] = (m, o, s)

# -----------------------------------------
 
    _logger = logging.getLogger('numina')
    _logger.setLevel(logging.DEBUG)

    nthreads = 1

    imagesl = []
    omasksl = []
    # Reading data:
    initd = []
    
    for i in pv['inputs']['images']:
        initd.append(image.EmirImage(datafile=i,
                                     maskfile=pv['inputs']['images'][i][0],
                                     offset=pv['inputs']['images'][i][1]))
        om = image.Image(datafile='bpm.fits')
    
    
    offsets = [im.offset for im in initd]
    
    finalshape, offsetsp = combine_shape((2048, 2048), offsets)
    
    for im, off in zip(initd, offsetsp):
        im.noffset = off
    
    imagesl.append(initd)
    
    # Initialize processing nodes, step 1
    dark_data = pyfits.getdata(pv['inputs']['master_dark'])    
    flat_data = pyfits.getdata(pv['inputs']['master_flat'])

    # basic corrections in the first pass
    flow1 = SerialFlow([OpenImage(mode='update'),
                         DarkCorrector(dark_data),
                         NonLinearityCorrector(pv['optional']['linearity']),
                         FlatFieldCorrector(flat_data),
                         #BackupNode(),
                         CloseImage(output_verify='fix')])
    
    _logger.info('Basic processing')
    #result = worker.para_map(procd['basic'], imagesl[0], nthreads=nthreads)
    
    result = map(flow1, imagesl[0])
    imagesl.append(result)
    
    flow2 = SerialFlow([OpenImage(), 
                        SaveAsNode(namegen=basic_naming),
                        ParallelFlow([CloseImage(),
                        SerialFlow([OpenImage(mode='update'),
                                    ResizeNode(finalshape),
                                    CloseImage(output_verify='fix')])
                                  ]),
                        OutputSelector(2, (1,))
                        ])
    
    result = map(flow2, imagesl[1])

    imagesl.append(result)
    
    iteration = 0
    
    # Creating object masks
    
    flow = SerialFlow([OpenImage(), 
                       CopyMask(omask_naming(iteration)),
                       ParallelFlow([CloseImage(), CloseImage()]),
                       OutputSelector(2, (1,))                              
                    ])
    
    result = map(flow, imagesl[2])
    omasksl.append(result)
    
    _logger.info('Iter %d, superflat correction (SF)', iteration)

    # Step 2, compute superflat
    _logger.info('Iter %d, SF: computing scale factors', iteration)
    # Actions:
    flow3 = SerialFlow([OpenImage(mode='readonly'),
                        AdaptorNode(compute_median, noutputs=2),                                                
                        ParallelFlow([IdNode(), CloseImage()]),
                        OutputSelector(2, (0,))                                        
                        ]
    )
    
    
    scales = map(flow3, imagesl[2])

    # Operation to create an intermediate sky flat
        
    result = map(OpenImage(), imagesl[2])
    _logger.info("Iter %d, SF: combining the images without offsets", iteration)
    
    
    sf_data = image.combine('median', result, scales=scales)
    map(CloseImage(), result)
    
    # Zero masks
    # TODO Do a better fix here
    # This is to avoid negative of zero values in the flat field
    mm = sf_data[0] <= 0
    sf_data[0][mm] = 1
    
    # We are saving here only data part
    pyfits.writeto('emir_sf.iter.%02d.fits' % iteration, sf_data[0], clobber=True)
    
    map(CloseImage(), result)
    #sys.exit(0)
    
    # Step 3, apply superflat
    _logger.info("Iter %d, SF: apply superflat", iteration)

    
    flow = SerialFlow([OpenImage(),
                        SaveAsNode(namegen=skyflat_naming(iteration)),
                        ParallelFlow([CloseImage(),
                                         SerialFlow([OpenImage(mode='update'),
                                                     FlatFieldCorrector(sf_data[0],
                                                                        mark=False,
                                                                        region=True),
                                                      CloseImage()])]),
                        OutputSelector(2, (1,))])

    result = map(flow, imagesl[2])    
    imagesl.append(result)
    
    _logger.info('Iter %d, sky correction (SC)', iteration)    
    
    _logger.info('Iter %d, SC: computing simple sky', iteration)
    
    # Compute simple sky
    # Actions:
    flow = SerialFlow([OpenImage(mode='readonly'),
                       AdaptorNode(compute_sky_simple, noutputs=2),
                       ParallelFlow([IdNode(),
                                     CloseImage()
                                    ]),
                       OutputSelector(2, (0,))
                    ]
    )
    
    
    
    skyback = map(flow, imagesl[3])
    sys.exit(0)
    result = map(OpenImage(), imagesl[3])
    _logger.info("Iter %d, Combining the images", iteration)
    
    # Write a node for this
    extinc = [pow(10, 0.4 * 1.2 * 0.01)  for i in result]
    sf_data = image.combine2('median', result, zeros=skyback, scales=extinc, dtype='float32')
    
    # We are saving here only data part
    pyfits.writeto('result.%02d.fits' % iteration, sf_data[0], clobber=True)
    
    map(CloseImage(), result)
    
    _logger.info('Iter %d, generating objects mask', iteration)
    sex_om = SextractorObjectMask(segmask_naming(iteration))
    
    
    print sf_data[0].shape
    print sf_data[0]
    
    obj_mask = sex_om(sf_data[0])
    
    _logger.info('Iter %d, merging object mask with masks', iteration)
    #map(lambda f: mask_merging(f, obj_mask, itern), self.images)
    # We need to store this result
    dflow = SerialFlow([OpenImage(mode='readonly'),
                        MaskMerging(obj_mask, omask_naming(iteration)),
                        CloseImage()
                        ])
    
    procd['mmerge'] = dflow
    
    result = map(procd['mmerge'], imagesl[3])
    
    _logger.info('Iter %d, finished', iteration)
    
    sys.exit(0)
