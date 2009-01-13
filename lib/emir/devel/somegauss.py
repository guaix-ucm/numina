
import numpy

sample = [[200.0,356.9,45.0,100.0], [1024.0,1024.0,9.0,10.0]]

def somegauss(shape,level, gaussians = None):
  '''Some gaussians in an image'''
  intensity = 10.0
  im = level * numpy.ones(shape)
  if gaussians is not None:
      y, x = numpy.indices(shape)
      for i in gaussians:
          x0 = i[0]
          y0 = i[1]
          sigma0 = i[2]
          intensity = i[3]
          im +=  intensity * numpy.exp(-((x - x0)**2+(y - y0)**2)/(sigma0)**2)
  nim = numpy.random.poisson(im)
  return nim
