============
Installation
============

Requirements
------------

Python >= 3.6 is required. Additionally the following packages are required
in order to work properly:

 - `setuptools <http://pythonhosted.org/setuptools/>`_
 - `numpy <http://numpy.scipy.org/>`_ 
 - `scipy <http://www.scipy.org>`_
 - `astropy <http://www.astropy.org>`_
 - `PyYaml <http://http://pyyaml.org/>`_
 - `matplotlib <https://matplotlib.org/>`_
 - `scikit-image <https://scikit-image.org/>`_
 - `lmfit <https://lmfit.github.io/lmfit-py/>`_
 - `python-dateutil <https://pypi.org/project/python-dateutil>`_

Cython is required if you build the code from the development repository:

 - `Cython <https://cython.org/>`_

The following packages are optional, for building documentation and testing:

 - `sphinx`_  to build the documentation
 - `pytest`_  for testing
 - `pytest-remotedata <https://github.com/astropy/pytest-remotedata>`_ for testing with remote datasets


Installing numina
-----------------

The preferred methods of installation are `pip <https://pip.pypa.io>`_ or
`conda <https://docs.conda.io/en/latest/>`__, using prebuilt packages.

Using pip
+++++++++

Run::

    pip install numina

Pip will download all the required dependencies and a precompiled versión of numina
(if it exists for your platform) from `PyPI <https://pypi.org/project/numina/>`__.

.. note:: If possible, pip will install a precompiled version of numina in wheel format.
            If such a version does not exist, pip will download and compile the source code.
            Numina has some portions of C and C++ code. You will need a C/C++ compiler
            such as ``gcc`` or ``clang`` (see :ref:`deploy_source` below)

Pip can install packages in different locations. You can use the ``--user`` option
to install packages in your home directory.

Our recomended option is to perform isolated installations
using virtualenv or venv. See :ref:`deploy_venv` for details.

.. warning:: Do not use ``sudo pip`` unless you *really really* know what you are doing.


Using conda
+++++++++++
Numina packages for conda are provided in the `conda-forge <https://conda-forge.org/>`_ channel. To install
the latest version of numina run::

    conda update -c conda-forge numina

See :ref:`deploy_conda` for details.


.. _deploy_source:

Building from source
--------------------

You may end up building from source if there is not a stable precompiled version
of numina for your platform or if you are doing development based on numina.

Prerequisites
+++++++++++++
You will need a compiler suite and the development headers of Python and Numpy.

If you are building the development version of numina, you will also need Cython
to translate Cython code into C/C++. If your source code is from a release,
the translated files are included, and hence do not require Cython.


The released sources of numina can be downloaded from PyPI. If you require instead
the development version, it can can be checked out with::

    git clone https://github.com/guaix-ucm/numina.git

Building and installing
++++++++++++++++++++++++

To build numina, run::

    python setup.py build

.. note:: In macOS Mojave, the compilation will fail unless the following
            environment variable is defined::

                export MACOSX_DEPLOYMENT_TARGET=10.9


To install numina, run::

    python setup.py install


If you get an error about insufficient permissions to install, you are probably trying to access
directories owned by root. Try instead::

    python setup.py install --user

or perform the installation inside an isolated environment, such as conda or venv.


.. warning:: Do not ``sudo python setup.py install`` unless you really really know what you are doing.


Building the documentation
--------------------------
The Numina documentation is base on `sphinx`_. With the package installed, the 
html documentation can be built from the `doc` directory::

  $ cd doc
  $ make html
  
The documentation will be copied to a directory under `build/sphinx`.
  
The documentation can be built in different formats. The complete list will appear
if you type `make` 
  
.. _virtualenv: https://virtualenv.pypa.io/
.. _sphinx: http://sphinx.pocoo.org
.. _pytest: http://pytest.org/latest/
.. _virtualenv_install: https://virtualenv.pypa.io/en/latest/installation.html
.. _virtualenv_usage: https://virtualenv.pypa.io/en/latest/userguide.html
