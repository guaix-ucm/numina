#!/usr/bin/env python

from distutils.core import setup, Extension
import os

# pkg-config code from https://trac.xiph.org/browser/icecast/trunk/shout-python/setup.py

ver = '0.1.0'

pkgcfg = os.popen('pkg-config --cflags emirdfp')
cflags = pkgcfg.readline().strip()
pkgcfg.close()
pkgcfg = os.popen('pkg-config --libs emirdfp')
libs = pkgcfg.readline().strip()
pkgcfg.close()

iflags = [x[2:] for x in cflags.split() if x[0:2] == '-I']
extra_cflags = [x for x in cflags.split() if x[0:2] != '-I']
libdirs = [x[2:] for x in libs.split() if x[0:2] == '-L']
libsonly = [x[2:] for x in libs.split() if x[0:2] == '-l']

libsonly.append('boost_python')

pyemir = Extension(name = 'emir', 
		sources = ['emir_wrap.cc'],
		include_dirs = iflags,
		extra_compile_args = extra_cflags,
		library_dirs = libdirs,
		libraries = libsonly)

setup(name='pyemir',
      version=ver,
      author='Sergio Pascual',
      author_email='spr@astrax.fis.ucm.es',
      url='http://guaix.fis.ucm.es',
      license='GPLv3',
      description = 'Bindings for EMIR Data Factory Pipeline',
      long_description = 'Bindings for EMIR Data Factory Pipeline',
      ext_modules=[pyemir])

