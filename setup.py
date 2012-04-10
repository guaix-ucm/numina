#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

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

setup(name='numina',
      version='0.5.0',
      author='Sergio Pascual',
      author_email='sergiopr@fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/emir',
      download_url='ftp://astrax.fis.ucm.es/pub/users/spr/emir/numina-0.5.0.tar.gz',
      license='GPLv3',
      description='Numina reduction package',
      packages=find_packages('.'),
      package_data={'numina': ['*.cfg'],
                    },
      ext_modules=[cext, uext],
      entry_points={
                      'console_scripts': ['numina = numina.user:main'],
                      },
      test_suite="nose.collector",
      install_requires=['setuptools', 'numpy', 'pyfits', 'scipy', 
		'sphinx', 'PyYaml', 'nose'],
      classifiers=[
                   "Programming Language :: Python",
                   'Development Status :: 3 - Alpha',
                   "Environment :: Other Environment",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   ],
      long_description='''\
      This is Numina reduction package
      
      Numina is the data reduction package used for the following GTC
      instruments: EMIR, FRIDA, MEGARA, MIRADAS
      
      Maintainer: sergiopr@fis.ucm.es
      ''',
      )
