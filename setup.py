#!/usr/bin/env python

from setuptools import setup, Extension
from setuptools import find_packages

import sys

try:
    import numpy
except ImportError:
    print 'numpy is required to install numina'
    sys.exit(1)

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

from numina import __version__

setup(name='numina',
      version=__version__,
      author='Sergio Pascual',
      author_email='sergiopr@fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/emir',
      download_url='ftp://astrax.fis.ucm.es/pub/software/numina/numina-%s.tar.gz' % __version__,
      license='GPLv3',
      description='Numina reduction package',
      packages=find_packages('.'),
      ext_modules=[cext, uext],
      entry_points={'console_scripts': ['numina = numina.user:main']},
      requires=REQUIRES,
      setup_requires=['numpy'],
      install_requires=REQUIRES,
      classifiers=[
                   "Programming Language :: Python :: 2.7",
                   'Development Status :: 3 - Alpha',
                   "Environment :: Other Environment",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   ],
      long_description=open('README.txt').read()
      )
