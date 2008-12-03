# $Id$

import time

def print_timing(func):
    '''Timing decorator'''
    def wrapper(*arg, **keywords):
        t1 = time.time()
        res = func(*arg, **keywords)
        t2 = time.time()
        print '%s took %0.3f s' % (func.func_name, (t2 - t1))
        return res
    return wrapper

def test():
    
    import _combine
    import numpy
    import numpy.random
    import pyfits

    @print_timing
    def ucombine(a, result, b, c, images, d, e, f):
        return _combine._combine(1, result, 3, 4, images, 6, 7, 8)

    def random_images(shape, nimages):    
        images = [numpy.random.standard_normal(shape) for i in range(nimages)]
        return images
    
    @print_timing
    def fits_images(filename, nimages):
        images = [pyfits.getdata(filename) for i in range(nimages)]
        return images
    
    shape = (2048, 2048)
    nimages = 100
    #images = random_images(shape, nimages)
    filename = '/home/inferis/spr/IR/apr21/apr21_0046.fits'
    images = fits_images(filename, nimages)
    
    result = numpy.ones(shape)
    print 'starting'
    ucombine(1, result, 3, 4, images, 6, 7, 8)
    print result
    
if __name__ == "__main__":
    test()