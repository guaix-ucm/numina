#!/usr/bin/env python

from __future__ import print_function

from setuptools import setup, Extension
from setuptools import find_packages

import sys

try:
    import numpy
except ImportError:
    print('numpy is required to install numina')
    sys.exit(1)

numpy_include = numpy.get_include()

ext1 = Extension('numina.array._combine',
                ['src/combinemodule.cc',
                 'src/operations.cc',
                 'src/nu_combine_methods.cc',
                 'src/nu_combine.cc'
                 ],
          include_dirs=[numpy_include])

ext2 = Extension('numina.array._ufunc',
                ['src/ufunc.cc',
                 ],
          include_dirs=[numpy_include])

ext3 = Extension('numina.array._nirproc', 
                 ['src/nirprocmodule.cc'],
                include_dirs=[numpy_include])

# requires is not used by pip
# but install_requires is not supported 
# http://bugs.python.org/issue1635217
REQUIRES = ['setuptools', 'numpy (>=1.6)', 'pyfits', 'scipy', 'PyYaml']
IREQUIRES = ['setuptools', 'numpy>=1.6', 'pyfits', 'scipy', 'PyYaml']

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
      ext_modules=[ext1, ext2, ext3],
      entry_points={'console_scripts': ['numina = numina.user:main']},
      requires=REQUIRES,
      setup_requires=['numpy'],
      install_requires=IREQUIRES,
      use_2to3 = True,
      test_suite= "numina.tests",
      classifiers=[
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3.0",
                   'Development Status :: 3 - Alpha',
                   "Environment :: Other Environment",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   ],
      long_description=open('README.txt').read()
      )
