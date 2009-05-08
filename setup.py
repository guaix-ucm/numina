#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

import numpy

numpy_include =  numpy.get_include()
ex1 = Extension('emir.image._combine',['src/combinemodule.cc', 'src/methods.cc'],
          include_dirs=[numpy_include])

setup(name='pyemir',
      version='0.0.1',
      author='Sergio Pascual',
      author_email='sergiopr@astrax.fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/emir',
      license='GPLv3',
      description='EMIR Data Processing Pipeline',
      long_description='EMIR Data Processing Pipeline',      
      packages=find_packages('lib'),
      package_dir={'': 'lib'},
      package_data = {'emir.simulation': ['*.dat'],
                      'numina': ['*.cfg']},
      ext_modules=[ex1],
      requires=['pyfits', 'scipy'],
      entry_points = {'console_scripts': ['numina = numina.user:main',]},
      # how to use nose
      #http://www.plope.com/Members/chrism/nose_setup_py_test
      #test_suite = "nose.collector",
      test_suite = "emir.tests",
      data_files = [('/etc/numina', ['numina/logging.ini'])]
      )
