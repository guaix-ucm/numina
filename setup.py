
from setuptools import setup, Extension
import numpy
from Cython.Distutils import build_ext


def get_extensions():
    numpy_include = numpy.get_include()

    extensions = []

    ext1 = Extension('numina.array._combine',
                     ['numina/array/src/combinemodule.cc',
                      'numina/array/src/operations.cc',
                      'numina/array/src/nu_combine_methods.cc',
                      'numina/array/src/nu_combine.cc'
                      ],
                     include_dirs=[numpy_include])

    extensions.append(ext1)

    ext3 = Extension('numina.array._nirproc',
                     ['numina/array/src/nirproc.pyx'],
                     include_dirs=[numpy_include],
                     language='c++'
                     )

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

    return extensions


if __name__ == '__main__':
    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=get_extensions()
    )
