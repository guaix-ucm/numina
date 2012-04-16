#!/usr/bin/env python

from setuptools import setup, Extension

import numpy

numpy_include = numpy.get_include()
cext = Extension('numina.array._combine',
                ['src/combinemodule.cc',
                 'src/operations.cc',
                 'src/nu_combine_methods.cc',
                 'src/nu_combine.cc'
                 ],
          include_dirs=[numpy_include])

uext = Extension('numina.array._ufunc',
                ['src/ufunc.cc',
                 ],
          include_dirs=[numpy_include])

REQUIRES = ['setuptools', 'numpy', 'pyfits', 'scipy', 'PyYaml']

setup(name='numina',
      version='0.6.0dev',
      author='Sergio Pascual',
      author_email='sergiopr@fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/emir',
      download_url='ftp://astrax.fis.ucm.es/pub/software/numina/numina-0.6.0dev.tar.gz',
      license='GPLv3',
      description='Numina reduction package',
      packages=['numina', 'numina.array', 'numina.flow', 
                'numina.frame', 'numina.frame.aperture',
                'numina.instrument',
                'numina.recipes', 'numina.serialize', 
                'numina.tests', 'numina.util'],
      ext_modules=[cext, uext],
      data_files=[('share/numina/pipelines', ['pipelines/README'])],
      scripts=['scripts/numina'],
      requires=REQUIRES,
      install_requires=REQUIRES,
      classifiers=[
                   "Programming Language :: Python",
                   'Development Status :: 3 - Alpha',
                   "Environment :: Other Environment",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   ],
      long_description=open('README.txt').read()
      )
