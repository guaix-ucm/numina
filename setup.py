
import numpy
from setuptools import setup, Extension


def get_extensions():
    numpy_include = numpy.get_include()

    extensions = []
    np_api_min = ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")

    ext1 = Extension('numina.array._combine',
                     ['src/numina/array/src/combinemodule.cc',
                      'src/numina/array/src/operations.cc',
                      'src/numina/array/src/nu_combine_methods.cc',
                      'src/numina/array/src/nu_combine.cc'
                      ],
                     define_macros=[np_api_min],
                     include_dirs=[numpy_include])

    extensions.append(ext1)

    ext3 = Extension('numina.array._nirproc',
                     ['src/numina/array/src/nirproc.pyx'],
                     define_macros=[np_api_min],
                     include_dirs=[numpy_include],
                     language='c++'
                     )

    extensions.append(ext3)

    ext4 = Extension('numina.array.trace._traces',
                     ['src/numina/array/trace/traces.pyx',
                      'src/numina/array/trace/Trace.cpp'],
                     define_macros=[np_api_min],
                     include_dirs=[numpy_include],
                     language='c++')
    extensions.append(ext4)

    ext5 = Extension('numina.array.trace._extract',
                     ['src/numina/array/trace/extract.pyx'],
                     define_macros=[np_api_min],
                     include_dirs=[numpy_include],
                     language='c++')
    extensions.append(ext5)

    ext6 = Extension('numina.array.peaks._kernels',
                     ['src/numina/array/peaks/kernels.pyx'],
                     define_macros=[np_api_min],
                     language='c')
    extensions.append(ext6)

    ext7 = Extension('numina.array._bpm',
                     ['src/numina/array/bpm.pyx'],
                     define_macros=[np_api_min],
                     include_dirs=[numpy_include],
                     language='c++')
    extensions.append(ext7)

    ext8 = Extension('numina.array._clippix',
                     ['src/numina/array/clippix.pyx'],
                     define_macros=[np_api_min],
                     include_dirs=[numpy_include],
                     language='c')
    extensions.append(ext8)

    return extensions


if __name__ == '__main__':
    setup(
        ext_modules=get_extensions()
    )
