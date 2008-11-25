#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name='pyemir',
      version='0.1.0',
      author='Sergio Pascual',
      author_email='spr@astrax.fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/emir',
      license='GPLv3',
      description='EMIR Data Processing Pipeline',
      long_description='EMIR Data Processing Pipeline',
      package_dir={'emir': 'src/emir'},
      packages=['emir', 'emir.image'],
      ext_modules=[Extension('emir.test',['lib/test.cc'],libraries=['boost_python'])],
      scripts=['imcombine.py'],
      requires=['pyfits', 'scipy'],
      )

