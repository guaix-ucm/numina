#!/usr/bin/env python
from __future__ import print_function
import sys
from numina import __version__
from setuptools import setup, Extension
from setuptools import find_packages

try:
    import numpy
except ImportError:
    sys.exit('numpy is required to install numina')

numpy_include = numpy.get_include()

ext = []
cmdclass = {}
file_extension = ""

try:
    from Cython.Distutils import build_ext
    file_extension = "pyx"
    cmdclass['build_ext'] = build_ext

except ImportError:
    print('We do not have Cython, just using the generated files')

    file_extension = "cpp"
    cmdclass = {}

ext.append(Extension('numina.array._nirproc',['src/nirproc.%s' %file_extension],language='c++'))
ext.append(Extension('numina.array.trace._traces',['numina/array/trace/traces.%s'%file_extension,'numina/array/trace/Trace.cpp'],language='c++'))
ext.append(Extension('numina.array.trace._extract',['numina/array/trace/extract.%s'%file_extension],language='c++'))
ext.append(Extension('numina.array._combine',['src/combinemodule.cc','src/operations.cc','src/nu_combine_methods.cc','src/nu_combine.cc'],))
ext.append(Extension('numina.array._ufunc',['src/ufunc.cc',],))
ext.append(Extension('numina.array.wavecal.arccalibration',['numina/array/wavecal/arccalibration.%s' %file_extension],language='c++'))

REQUIRES = ['setuptools', 'six>=1.7', 'numpy>=1.7', 'astropy>=1.0', 'scipy', 'PyYaml']

# Some packages are required only in some versions of Python

# In versions >= 2.7 and < 3.4, we require singledispatch
# In 3.4 onwards, its in stdlib
if sys.hexversion < 0x3040000:
    REQUIRES += ['singledispatch']

setup(name='numina',
      version=__version__,
      include_dirs = [numpy.get_include()],
      url='http://guaix.fis.ucm.es/projects/numina',
      license='GPLv3',
      description='Numina reduction package',
      packages=find_packages('.'),
      package_data={'numina.tests.drps.1': ['drp.yaml'],},
      ext_modules=ext,
      entry_points={'console_scripts': ['numina = numina.user.cli:main',],
                    'numina.storage.1': ['numina_default = numina.store.default:load_cli_storage',],
                    },
      setup_requires=['numpy'],
      install_requires=REQUIRES,
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass=cmdclass,
      classifiers=[
                   "Programming Language :: C",
                   "Programming Language :: Cython",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3.3",
                   "Programming Language :: Python :: 3.4",
                   "Programming Language :: Python :: Implementation :: CPython",
                   'Development Status :: 3 - Alpha',
                   "Environment :: Other Environment",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   ],
      long_description=open('README.rst').read()
      )
