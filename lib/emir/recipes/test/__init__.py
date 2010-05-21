
import logging
import time 
import sys
import os

import numpy

import worker
import image
from node import SerialNode, AdaptorNode, ParallelAdaptor, IdNode
from processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from processing import OpenNode, CloseNode, BackupNode
from processing import compute_median
from utils import iterqueue

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger("numina")

if __name__ == '__main__':
    import pyfits

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

# Initialize processing nodes, step 1
    dark_data = pyfits.getdata(pv['inputs']['master_dark'])    
    flat_data = pyfits.getdata(pv['inputs']['master_flat'])
        
    corrector1 = DarkCorrector(dark_data)
    corrector2 = NonLinearityCorrector(pv['optional']['linearity'])
    corrector3 = FlatFieldCorrector(flat_data)
        
    

    # Reading data:
    initd = []

    for i in pv['inputs']['images']:
        initd.append(image.EmirImage(datafile=i, maskfile=pv['inputs']['images'][i][0]))

    snode1 = SerialNode([OpenNode(), corrector1, corrector2, corrector3, BackupNode(), CloseNode()])
    _logger.info('Basic processing')
    store1 = worker.para_map(snode1, initd, nthreads=4)
    
    
    iteration = 0
    
    _logger.info('Iter %d, superflat correction (SF)', iteration)

    # Step 2, compute superflat
    _logger.info('Iter %d, SF: computing scale factor', iteration)
    snode2 = SerialNode([OpenNode(), AdaptorNode(compute_median), 
                         ParallelAdaptor(IdNode(), CloseNode())])
    store2 = worker.para_map(snode2, store1, nthreads=4)

    # Operation to create an intermediate sky flat
    map(OpenNode(), store1)
    scales = [s for (s,) in store2]
    _logger.info("Iter %d, SF: combining the images without offsets", iteration)
    sf_data = image.combine('median', store1, scales=scales)
    map(CloseNode(), store1)

    # We are saving here only data part
    pyfits.writeto('emir_sf.iter.%02d.fits' % iteration, sf_data[0], clobber=True)

    sys.exit(0)
    # Step 3, apply superflat
    snode3 = SerialNode([OpenNode(), FlatFieldCorrector(sf_data, mark=False), CloseNode()])

    store3 = worker.para_map(snode3, store1, nthreads=4)

    # We need to store this result

    sys.exit(0)


