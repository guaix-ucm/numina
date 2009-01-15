#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include =  numpy.get_include()
ex1 = Extension('emir.image._combine',['src/combinemodule.c'],
          include_dirs=[numpy_include])

setup(name='pyemir',
      version='0.1.0',
      author='Sergio Pascual',
      author_email='spr@astrax.fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/emir',
      license='GPLv3',
      description='EMIR Data Processing Pipeline',
      long_description='EMIR Data Processing Pipeline',
      package_dir={'emir': 'lib/emir'},
      packages=['emir', 'emir.image', 'emir.devel',
                 'emir.simulation', 'emir.tests'],
      ext_modules=[ex1],
      scripts=['imcombine.py'],
      requires=['pyfits', 'scipy'],
      )
