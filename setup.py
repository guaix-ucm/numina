#!/usr/bin/env python

from __future__ import print_function

from setuptools import setup, Extension
from setuptools import find_packages

import sys

try:
    import numpy
except ImportError:
    sys.exit('numpy is required to install numina')

numpy_include = numpy.get_include()

# try to handle gracefully Cython
try:
    from Cython.Distutils import build_ext
    ext3 = Extension('numina.array._nirproc', 
                 ['src/nirproc.pyx'],
                include_dirs=[numpy_include],
                language='c++')
    cmdclass = {'build_ext': build_ext}
except ImportError:
    print('We do not have Cython, just using the generated files')
    ext3 = Extension('numina.array._nirproc', 
                 ['src/nirproc.cpp'],
                include_dirs=[numpy_include])
    cmdclass = {}


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

REQUIRES = ['setuptools', 'numpy>=1.6', 'astropy>=0.4', 'scipy', 'PyYaml']

from numina import __version__

setup(name='numina',
      version=__version__,
      author='Sergio Pascual',
      author_email='sergiopr@fis.ucm.es',
      url='http://guaix.fis.ucm.es/projects/numina',
      license='GPLv3',
      description='Numina reduction package',
      packages=find_packages('.'),
      package_data={'numina.tests.drps.1': ['drp.yaml'],
                   },
      ext_modules=[ext1, ext2, ext3],
      entry_points={'console_scripts': ['numina = numina.user:main']},
      setup_requires=['numpy'],
      install_requires=REQUIRES,
      use_2to3=True,
      zip_safe=False,
      test_suite= "numina.tests",
      cmdclass=cmdclass,
      classifiers=[
                   "Programming Language :: C",
                   "Programming Language :: Cython",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3.0",
                   "Programming Language :: Python :: Implementation :: CPython",
                   'Development Status :: 3 - Alpha',
                   "Environment :: Other Environment",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   ],
      long_description=open('README.txt').read()
      )
