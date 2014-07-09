#
# Copyright 2008-2014 Universidad Complutense de Madrid
# 
# This file is part of Numina
# 
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
# 

'''
Results of the Observing Blocks 
'''

from astropy.io import fits

from .dataframe import DataFrame

class ObservationResult(object):
    '''The result of a observing block.
    
    '''
    def __init__(self, mode=None):
        self.id = 1
        self.mode = mode
        self.instrument = None
        self.frames = [] 
        self.children = [] # other ObservationResult
        self.pipeline = 'default'

def dataframe_from_list(values):
    '''Build a DataFrame object from a list.'''
    if(isinstance(values, basestring)):
        return DataFrame(filename=values)
    elif(isinstance(values, fits.HDUList)):
        return DataFrame(frame=values)
    else:
        # FIXME: modify when format is changed
        # For this format
        return DataFrame(filename=values[0], itype=values[1])

def obsres_from_dict(values):
    '''Build a ObservationResult object from a dictionary.'''
    obsres = ObservationResult()
    
    obsres.id = values.get('id', 1)
    obsres.mode = values['mode']
    obsres.instrument = values['instrument']
    obsres.configuration = values.get('configuration', 'default')
    obsres.pipeline = values.get('pipeline', 'default')
    obsres.frames = [dataframe_from_list(val) for val in values['frames']]
    
    return obsres

# We are not using these two for the moment

def frameinfo_from_list(values):
    '''Build a FrameInformation object from a list.'''
    frameinfo = FrameInformation()
    if(isinstance(values, basestring)):
        frameinfo.label = values
        frameinfo.itype = 'UNKNOWN'
    else:
        # FIXME: modify when format is changed
        # For this format        
        frameinfo.label = values[0]
        frameinfo.itype = values[1]
    return frameinfo

class FrameInformation(object):
    '''Information of a frame observed during a block.'''
    def __init__(self):
        self.label = None
        self.itype = None

    def __repr__(self):
        return 'FrameInformation(label=%r, itype=%r)' % (self.label, self.itype)
