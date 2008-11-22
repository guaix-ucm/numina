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

def basecombine(input, offsets=None, masks=None):
    
    # Check length of offsets
    if offsets is not None and len(input) != len(offsets):
        # Here we should have and error condition
        raise Error('Error, number of images not equal to the number of offsets')
    
    if masks is not None and len(input) != len(masks):
        # Here we should have and error condition
        raise Error('Error, number of images not equal to the number of mask')
    
    
    # Computing the final size of the images
    sizes = [i.shape for i in input]        
    tpixel =  map(lambda x,y: (x[0] + y[0] , x[1] + y[1]) , sizes, offsets)
    
    min_of_offsets =(min(map(lambda x: x[0], offsets)), min(map(lambda x: x[1], offsets)))
    max_of_offsets =(max(map(lambda x: x[0], tpixel)), max(map(lambda x: x[1], tpixel)))
    
    final_size = (max_of_offsets[0] - min_of_offsets[0],
                  max_of_offsets[1] - min_of_offsets[1])
        
    # Result
    result = 0
    # Variance
    variance = 0
    # Number of images combined
    number = numpy.zeros(final_size)
    
    # Compute here
    
    return (result,variance,number)
  

def combine(input, method, offsets=None, masks=None, output=None):
    ''' Input is a list of arrays
        Method is a callable object that takes a list of values and returns a 3-tuple of values
        Offsets is a list of tuples
        Returns a 3-tuple of arrays
        '''    
    
    # Compute the size of the final image
    # No offsets, so the final image is equal to the first
    
    # Result
    result = input[0] * 0
    # Variance
    variance = input[0] * 0
    # Number of images combined
    number = input[0] * 0
    
    # Compute here
    for (index,val) in numpy.ndenumerate(result):
        values = [i[index] for i in input]
        (result[index], variance[index], number[index]) = method(values)
    
    return (result,variance,number)

if __name__ == "__main__":
    import numpy
    from combine_methods import *
    import time
    
    mymean = Mean()
    number = 5
    size = (2048,2048)
    array = numpy.ones(size[0]*size[1])
    array.shape = size
    
    input = [array.copy() for i in range(number)]
    offsets = [(3,2), (-5, -5), (-2,3), (5,5), (2,4)]
    
    #a = basecombine(input, offsets)
    #print a[2].shape
    print 'starting'
    t1 = time.time()
    a = combine(input, mymean)
    t2 = time.time()
    #
    print '%s took %0.3f s' % ('combine', (t2-t1))