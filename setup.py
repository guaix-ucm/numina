
from __future__ import print_function

from setuptools import setup, Extension
from setuptools import find_packages

import sys


def generate_extensions():

    try:
        import numpy
    except ImportError:
        sys.exit('numpy is required to build numina extensions')

    numpy_include = numpy.get_include()

    extensions = []

    ext1 = Extension('numina.array._combine',
                     ['src/combinemodule.cc',
                      'src/operations.cc',
                      'src/nu_combine_methods.cc',
                      'src/nu_combine.cc'
                      ],
                     include_dirs=[numpy_include])

    extensions.append(ext1)

    ext2 = Extension('numina.array._ufunc',
                     ['src/ufunc.cc',
                      ],
                     include_dirs=[numpy_include])

    extensions.append(ext2)

    # try to handle gracefully Cython
    try:
        from Cython.Distutils import build_ext
        cmdclass = {'build_ext': build_ext}

        ext3 = Extension('numina.array._nirproc',
                        ['src/nirproc.pyx'],
                        include_dirs=[numpy_include],
                        language='c++')

        extensions.append(ext3)
        ext4 = Extension('numina.array.trace._traces',
                         ['numina/array/trace/traces.pyx',
                          'numina/array/trace/Trace.cpp'],
                         include_dirs=[numpy_include],
                         language='c++')
        extensions.append(ext4)
        ext5 = Extension('numina.array.trace._extract',
                         ['numina/array/trace/extract.pyx'],
                         include_dirs=[numpy_include],
                         language='c++')
        extensions.append(ext5)
        ext6 = Extension('numina.array.peaks._kernels',
                         ['numina/array/peaks/kernels.pyx'],
                         language='c')
        extensions.append(ext6)
        ext7 = Extension('numina.array._bpm',
                         ['numina/array/bpm.pyx'],
                         include_dirs=[numpy_include],
                         language='c++')
        extensions.append(ext7)

    except ImportError:
        cmdclass = {}
        print('We do not have Cython, just using the generated files')
        ext3 = Extension('numina.array._nirproc',
                     ['src/nirproc.cpp'],
                    include_dirs=[numpy_include])
        extensions.append(ext3)
        ext4 = Extension('numina.array.trace._traces',
                         ['numina/array/trace/traces.cpp',
                          'numina/array/trace/Trace.cpp'],
                         include_dirs=[numpy_include],
                         language='c++')
        extensions.append(ext4)
        ext5 = Extension('numina.array.trace._extract',
                         ['numina/array/trace/extract.cpp'],
                         include_dirs=[numpy_include],
                         language='c++')
        extensions.append(ext5)
        ext6 = Extension('numina.array.peaks._kernels',
                         ['numina/array/peaks/kernels.c'],
                         language='c')
        extensions.append(ext6)
        ext7 = Extension('numina.array._bpm',
                         ['numina/array/bpm.cpp'],
                         include_dirs=[numpy_include],
                         language='c++')
        extensions.append(ext7)

    return extensions, cmdclass


def setup_package():

    from numina import __version__
    REQUIRES = ['setuptools', 'six>=1.7', 'numpy>=1.7', 'astropy>=1.3', 'scipy', 'PyYaml', 'matplotlib']

    META_DATA = dict(
        name='numina',
        version=__version__,
        author='Sergio Pascual',
        author_email='sergiopr@fis.ucm.es',
        url='http://guaix.fis.ucm.es/projects/numina',
        license='GPLv3',
        description='Numina reduction package',
        packages=find_packages('.'),
        package_data={
            'numina.drps.tests': [
                'drptest1.yaml',
                'drptest2.yaml',
                'drptest3.yaml',
                'drpclodia.yaml',
            ],
            'numina.drps.tests.configs': [
                'instrument-default.json'
            ],
        },
        #ext_modules=[ext1, ext2, ext3, ext4, ext5, ext6, ext7],
        entry_points={
            'console_scripts': [
                'numina = numina.user.cli:main',
                'numina-bpm = numina.array.bpm:main',
                'numina-imath = tools.imath:main',
                'numina-wavecalib = numina.array.wavecalib.__main__:main',
                'numina-ximshow = numina.array.display.ximshow:main',
                'numina-ximplotxy = numina.array.display.ximplotxy:main',
            ],
            },
        setup_requires=['numpy'],
        tests_require=['pytest'],
        install_requires=REQUIRES,
        zip_safe=False,
        classifiers=[
            "Programming Language :: C",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
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

    # For actins line egg_info and --version NumPy is not required
    if '--help' in sys.argv[1:] or \
      sys.argv[1] in ('--help-commands', 'egg_info', '--version'):
        pass
    else:
        # Generate extensions
        extensions, cmd_class = generate_extensions()

        META_DATA["cmdclass"] = cmd_class
        META_DATA["ext_modules"] = extensions

    setup(**META_DATA)


if __name__ == '__main__':
    setup_package()