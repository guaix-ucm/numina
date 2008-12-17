# $Id$

import time
def print_timing(func):
    '''Timing decorator'''
    def wrapper(*arg, **keywords):
        print 'starting %s' % (func.func_name,)
        t1 = time.time()
        res = func(*arg, **keywords)
        t2 = time.time()
        print '%s took %0.3f s' % (func.func_name, (t2 - t1))
        return res
    return wrapper
    
def test_conversion():
    import _combine
    import numpy
    
    def fun(data):
        '''Data is a list of values'''
        l = len(data)
        if l == 0:
            return (0., 1., 0)
        if l == 1:
            return (data[0], 0., 1)
        
        s = float(sum(data))
        s2 = float(sum(i * i for i in data))
        v = s2 / (l - 1) - (s * s) / (l * (l - 1));
        return (s / l, v, l)
    
    @print_timing
    def ucombine1(method, images, masks, out = None):
        return _combine.test1(method, images, masks)
    
    @print_timing
    def ucombine2(images, masks):
        return _combine.test2(images, masks)
    
    shape = (2048, 2048)
    num = 100
    images = [numpy.ones(shape) * i  for i in xrange(num)]
    out = numpy.zeros(shape)
    mask = numpy.ones(shape, dtype='bool')
    out1 = ucombine1(fun, images, [mask] * num)
    out1 = ucombine1(_combine.method1, images, [mask] * num)    
    out2 = ucombine2(images, [mask] * num)
    
    

def test():
    
    
    import _combine
    import numpy
    import numpy.random
    import pyfits

    @print_timing
    def ucombine(method, result, variance, number, images, shapes, masks, f):
        return _combine.test2(method, result, variance, number, images, shapes, masks, f)
    
    @print_timing
    def random_images(shape, nimages):    
        images = [numpy.random.standard_normal(shape) for i in range(nimages)]
        return images
    
    
    def generate_masks(shape, nimages):    
        images = [numpy.array(shape, dtype='bool') for i in range(nimages)]
        return images
    
    @print_timing
    def fits_images(filename, nimages):
        images = [pyfits.getdata(filename) for i in range(nimages)]
        return images
    
    @print_timing
    def constant_images(shape, nimages, value):
        a = numpy.zeros(shape)
        a += value
        return [a.copy() for i in xrange(nimages)]
    
    import threading
    
    class TRandom(threading.Thread):
        def __init__(self, list, shape, nimages, s):
            self.list = list
            self.shape = shape
            self.nimages = nimages
            self.s = s
            threading.Thread.__init__(self)
        def run(self):
            while True:
                image = numpy.random.standard_normal(shape)
                
                self.s.acquire()
                if len(self.list) >= self.nimages:
                    self.s.release()
                    return
                else:
                    self.list.append(image)
                    self.s.release()
    
    @print_timing
    def thread_random_images(shape, nimages, nthreads=4):
        result = []
        s = threading.Semaphore()
        
        threads = []
        for i in range(nthreads):
            a = TRandom(result, shape, nimages, s)
            threads.append(a)
        
        for a in threads:
            a.start()
        
        for a in threads:
            a.join()
        
        return result
    
    
    shape = (2048, 2048)
    nimages = 70
    
    #images = random_images(shape, nimages)
    #images = thread_random_images(shape, nimages, nthreads=8)
    
    #filename = '/home/inferis/spr/IR/apr21/apr21_0046.fits'
    #images = fits_images(filename, nimages)
    images = constant_images(shape, nimages, 23.4)
    masks = generate_masks(shape, nimages)
    
    result = numpy.ones(shape)
    variance = numpy.ones(shape)
    number = numpy.ones(shape, dtype='int')
    print len(masks)
    a = ucombine(1, result, variance, number, images, masks, masks, 8)
    print a[2][0, 0]
    
if __name__ == "__main__":
    test_conversion()
