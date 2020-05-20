
from __future__ import print_function

from setuptools import setup, Extension

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
        ext8 = Extension('numina.array._clippix',
                         ['numina/array/clippix.pyx'],
                         include_dirs=[numpy_include],
                         language='c')
        extensions.append(ext8)

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
        ext8 = Extension('numina.array._clippix',
                         ['numina/array/clippix.c'],
                         include_dirs=[numpy_include],
                         language='c')
        extensions.append(ext8)

    return extensions, cmdclass


def setup_package():

    META_DATA = dict(
        setup_requires=['numpy'],
        tests_require=['pytest', 'pytest-remotedata'],
        zip_safe=False,
        )

    # For actions like "egg_info" and "--version", numpy is not required
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
