============
Installation
============

Requirements
------------

Python >= 3.11 is required. pip installs the other required packages, among
them:

 - `numpy <https://numpy.org/>`_ 2 or later
 - `scipy <https://scipy.org>`_
 - `astropy <https://www.astropy.org>`_ 6.1.7 or later
 - `PyYaml <https://pyyaml.org/>`_
 - `matplotlib <https://matplotlib.org/>`_
 - `scikit-image <https://scikit-image.org/>`_
 - `lmfit <https://lmfit.github.io/lmfit-py/>`_
 - `reproject <https://reproject.readthedocs.io/en/stable/>`_
 - `ccdproc <https://ccdproc.readthedocs.io/>`_
 - `teareduce <https://pypi.org/project/teareduce/>`_
 - `jsonschema <https://python-jsonschema.readthedocs.io/>`_
 - `python-dateutil <https://pypi.org/project/python-dateutil>`_

The complete list is in the file ``pyproject.toml`` of the source code.


Installing numina
-----------------

The preferred method of installation is `pip <https://pip.pypa.io>`_, using
the prebuilt packages of `PyPI <https://pypi.org/project/numina/>`__, in a
virtual environment or in a `conda <https://docs.conda.io/en/latest/>`__
environment. There are prebuilt packages of numina for Linux and macOS, in
x86_64 and arm64.

Using pip
+++++++++

Run::

    pip install numina

Pip will download all the required dependencies and a precompiled versión of numina
(if it exists for your platform) from `PyPI <https://pypi.org/project/numina/>`__.

The graphical tools (such as ``numina-ximshow`` or ``numina-ximplotxy``) open
interactive windows using matplotlib, which requires a graphical backend.
To install the Qt bindings (`PyQt6 <https://pypi.org/project/PyQt6/>`_) run::

    pip install "numina[gui]"

Without Qt, matplotlib uses Tk if ``tkinter`` is available, but some options
(such as ``--geometry``) only work with Qt. Without any graphical backend
no windows are shown.

.. note:: If possible, pip will install a precompiled version of numina in wheel format.
            If such a version does not exist, pip will download and compile the source code.
            Numina has some portions of C and C++ code. You will need a C/C++ compiler
            such as ``gcc`` or ``clang`` (see :ref:`deploy_source` below)

Pip can install packages in different locations. You can use the ``--user`` option
to install packages in your home directory.

Our recommended option is to perform isolated installations
using virtualenv or venv. See :ref:`deploy_venv` for details.

.. warning:: Do not use ``sudo pip`` unless you *really really* know what you are doing.


Using conda
+++++++++++
Create and activate a conda environment, and install numina in it with pip::

    conda create --name numinaenv python=3.12
    conda activate numinaenv
    pip install numina

The packages of numina in the `conda-forge <https://conda-forge.org/>`_ channel
are no longer updated: they are older versions, and some dependencies of
numina, such as teareduce, are not in conda-forge.

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
to translate Cython code into C/C++.

The development version can can be checked out with::

    git clone https://github.com/guaix-ucm/numina.git

Building and installing
++++++++++++++++++++++++

To build and install numina, run::

    pip install .

You can all install the package in "editable" mode, including the "-e" option::

    pip install -e .

Running the tests
+++++++++++++++++

The tests use `pytest`_. Install numina with the optional dependencies
for testing and run pytest from the root of the source tree::

    pip install -e ".[test]"
    pytest

The tests that need data downloaded from a server are skipped by default,
they are enabled with ``--remote-data=any``. The slow tests, marked with
``slow``, are skipped by default too, and the tests of the continuous
integration run them. They are enabled with ``--runslow``::

    pytest --runslow

The tests can be run in parallel with `pytest-xdist
<https://pytest-xdist.readthedocs.io>`_; as they are short, a few
processes are faster than one for each CPU::

    pytest -n 8


.. _virtualenv: https://virtualenv.pypa.io/
.. _sphinx: http://sphinx.pocoo.org
.. _pytest: http://pytest.org/latest/
.. _virtualenv_install: https://virtualenv.pypa.io/en/latest/installation.html
.. _virtualenv_usage: https://virtualenv.pypa.io/en/latest/userguide.html
