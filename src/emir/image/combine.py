#
# Copyright 2008 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyMilia is free software: you can redistribute it and/or modify
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

# $Id$

from emir.exceptions import Error
from emir.decorators import print_timing

class NoMasks():
    '''An iterable dummy object that always return True'''
    def __iter__(self):
        return self
    def next(self):
        return self
    def __getitem__(self, wich):
        return True


class NoOffsets:
    '''An iterable dummy object that always return (0,0)'''
    def __iter__(self):
        return self
    def next(self):
        return self
    def __getitem__(self, wich):
        return (0, 0)

def offset_index(index, offset):
    return (index[0] - offset[0], index[1] - offset[1])

def valid_index(shape, index):
    '''Checks if the index is valid'''
    # import operator
    #reduce(operator.or_,map(lambda x:x<0,index))
    if (index[0] < 0) or (index[1] < 0):
        return False
    if (index[0] >= shape[0]) or (index[1] >= shape[1]):
        return False

    return True

@print_timing
def combine(input, method, offsets=None, masks=None, output=None):
    ''' Input is a list of arrays
        Method is a callable object that takes a list of values and returns a 3-tuple of values
        Offsets is a list of tuples
        Returns a 3-tuple of arrays
        '''    

    if offsets is not None and len(input) != len(offsets):        
        raise Error('Error, number of images not equal to the number of offsets')
    
    if masks is not None and len(input) != len(masks):        
        raise Error('Error, number of images not equal to the number of mask')
    
    if offsets is None:
        offsets = [(0, 0)] * len(input)
    
    # Computing the final size of the images    
    min_of_offsets = (min(map(lambda x: x[0], offsets)), min(map(lambda x: x[1], offsets)))
    
    shapes = [i.shape for i in input]        
    tpixel = map(lambda x, y: (x[0] + y[0] , x[1] + y[1]), shapes, offsets)

    max_of_offsets = (max(map(lambda x: x[0], tpixel)), max(map(lambda x: x[1], tpixel)))
    
    final_size = (max_of_offsets[0] - min_of_offsets[0],
                  max_of_offsets[1] - min_of_offsets[1])
    noffsets = [(i[0] - min_of_offsets[0], i[1] - min_of_offsets[1]) for i in offsets]
    
    # Result
    result = numpy.zeros(final_size)
    # Variance
    variance = numpy.zeros(final_size)
    # Number of images combined
    number = numpy.zeros(final_size)
    
    if masks is None:
        masks = NoMasks()

    #return internalc_simple(method, result, variance, number, input)
    return internalc_complex(method, result, variance, number, input, shapes, masks, noffsets)

@print_timing
def internalc_simple(method, result, variance, number, input, *therest):    
    for (index, val) in numpy.ndenumerate(result):
        # Combine without masks or offsets       
        values = [i[index] for i in input]                
        (result[index], variance[index], number[index]) = method(values)
    t2 = time.time()
    return (result, variance, number)

@print_timing
def internalc_complex(method, result, variance, number, input, shapes, masks, new_offsets):
    zinput = zip(input, shapes, masks, new_offsets)
    for (index, val) in numpy.ndenumerate(result):
        values = []
        # Compute the valid indices
        for (rimage, rshape, rmask, roffset) in zinput:            
            rindex = offset_index(index, roffset)
            if valid_index(index=rindex, shape=rshape) and rmask[rindex]:
                values.append(rimage[rindex])
                        
        (result[index], variance[index], number[index]) = method(values)
    return (result, variance, number)    


    

  
if __name__ == "__main__":
    
    import numpy
    from emir.image.combine_methods import *

    def main():
        mymean = Mean()
        number = 100000000 / 100
        number = 10
        size = (2048, 2048)
        array = numpy.ones(size[0] * size[1])
        array.shape = size
        mask = array > 0
        mask[0, 1] = False
        
        input = [array.copy() for i in range(number)]
        masks = [mask.copy() for i in range(number)]
        offsets = [(3, 2), (- 5, - 5), (- 2, 3), (5, 5), (2, 4)]
        offsets = offsets * (number / 5)
        masks = None
        offsets = None
        
        a = combine(input, mymean, masks=masks, offsets=offsets)
        
    import profile
    #profile.run('main()')
    main()