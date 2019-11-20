
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

    from numina import __version__

    META_DATA = dict(
        name='numina',
        version=__version__,
        author='Sergio Pascual',
        author_email='sergiopr@fis.ucm.es',
        url='https://github.com/guaix-ucm/numina',
        license='GPLv3',
        description='Astronomy data reduction library',
        packages=find_packages('.'),
        package_data={
            'numina.drps.tests': [
                'drptest1.yaml',
                'drptest2.yaml',
                'drptest3.yaml',
                'drptest4.yaml',
                'drpclodia.yaml',
            ],
            'numina.drps.tests.configs': [
                'instrument-*.json',
                'component-*.json',
                'properties-*.json',
                'setup-*.json'
            ],
        },
        #ext_modules=[ext1, ext2, ext3, ext4, ext5, ext6, ext7],
        entry_points={
            'console_scripts': [
                'numina = numina.user.cli:main',
                'numina-apply_integer_offsets = numina.array.wavecalib.apply_integer_offsets:main',
                'numina-bpm = numina.array.bpm:main',
                'numina-check_wlcalib = numina.array.wavecalib.check_wlcalib:main',
                'numina-imath = numina.tools.imath:main',
                'numina-r6-addnf = numina.tools.r6_addnf:main',
                'numina-r6-imcombine = numina.tools.r6_imcombine:main',
                'numina-r6-insert_keyword = numina.tools.r6_insert_keyword:main',
                'numina-r6-replace_image = numina.tools.r6_replace_image:main',
                'numina-wavecalib = numina.array.wavecalib.__main__:main',
                'numina-ximshow = numina.array.display.ximshow:main',
                'numina-ximplotxy = numina.array.display.ximplotxy:main',
            ],
            },
        setup_requires=['numpy'],
        tests_require=['pytest', 'pytest-remotedata'],
        install_requires=[
            'setuptools>=36.2.1',
            'six>=1.7',
            'numpy',
            'astropy>=2',
            'scipy>=0.19', 'PyYaml',
            'matplotlib',
            'enum34;python_version<"3.4"',
            'contextlib2;python_version<"3.5"',
            'python-dateutil', 'lmfit', 'scikit-image'
        ],
        zip_safe=False,
        classifiers=[
            "Programming Language :: C",
            "Programming Language :: C++",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            'Development Status :: 3 - Alpha',
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License (GPL)",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Software Development :: Libraries :: Application Frameworks",
            ],
        long_description=open('README.rst').read()
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
