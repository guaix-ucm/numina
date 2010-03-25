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

'''Recipe for image simulation'''


import logging

import numpy as np

import numina.recipes as nr
import numina.diskstorage as ds
import numina.qa as qa
from numina.simulation import RunCounter
from emir.instrument.detector import EmirDetector
from emir.instrument.headers import EmirImage

_logger = logging.getLogger("emir.recipes")

class ParameterDescription(nr.ParameterDescription):
    def __init__(self):
        inputs={'detector': {'shape': (2048, 2048),
                             'ron': 2.16,
                             'dark': 0.37,
                             'gain': 3.028,
                             'flat': 1.0,
                             'well': 65536,},
                'readout': {'mode': 'cds',
                            'reads': 3,
                            'repeat': 1,
                            'scheme': 'perline',
                            'exposure': 0},
                'nformat': "r%05d",
                }
        optional={}
        super(ParameterDescription, self).__init__(inputs, optional)

class Result(nr.RecipeResult):
    '''Result of the imaging mode recipe.'''
    def __init__(self, qa, result, name):
        super(Result, self).__init__(qa)
        self.products['result'] = result
        self.name = name
        
@ds.register(Result)
def _store(obj, where):
    ds.store(obj.products['result'], obj.name)   


class Recipe(nr.RecipeBase):
    '''Recipe to simulate EMIR images.
    '''
    def __init__(self):
        super(Recipe, self).__init__()
        #
        self.detector = None
        self.input = None
        _logger.info('FITS builder created')
        self.creator = EmirImage()
        _logger.info('Run counter created')
        self.runcounter = RunCounter("r%05d")
    
    def setup(self, param):
        super(Recipe, self).setup(param)
        _logger.info('Creating detector')
        
        self.detector = EmirDetector(**self.inputs['detector'])
        _logger.info('Configuring detector')
        self.detector.configure(self.inputs['readout'])
        
        self.input = np.zeros(self.inputs['detector']['shape'])
        self.detector.exposure(self.inputs['readout']['exposure'])
        self.repeat = self.inputs['readout']['repeat']
     
    def process(self):
        _logger.info('Creating simulated array')    
        output = self.detector.lpath(self.input)
        run, cfile = self.runcounter.runstring()
        headers = {'RUN': run}
        
        _logger.info('Collecting detector metadata')
        headers.update(self.detector.metadata())
        
        _logger.info('Building FITS structure')
        hdulist = self.creator.create(output, headers)
        return Result(qa.UNKNOWN, hdulist, cfile)

    def cleanup(self):
        pass

if __name__ == '__main__':
    import os
    import simplejson as json
    import tempfile
    
    from numina.user import main
    from numina.recipes import Parameters
    from numina.jsonserializer import to_json
      
    inputs={'detector': {'shape': (2048, 2048),
                             'ron': 2.16,
                             'dark': 0.37,
                             'gain': 3.028,
                             'flat': 1.0,
                             'well': 65536,},
                'readout': {'mode': 'cds',
                            'reads': 3,
                            'repeat': 1,
                            'scheme': 'perline',
                            'exposure': 0},
            'nformat': "r%05d",
                }
    optional={}
    
    p = Parameters(inputs, optional)
    
    tmpdir = tempfile.mkdtemp(suffix='emir')
    os.chdir(tmpdir)
    print 'Working directory is %s' % tmpdir
    
    cfile = 'config.json'
    
    f = open(cfile, 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
            
    main(['--run', 'simulate_image', cfile])
        


