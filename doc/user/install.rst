============
Installation
============

Requirements
------------

Python >= 3.9 is required. Additionally the following packages are required
in order to work properly:

 - `numpy <http://numpy.scipy.org/>`_ 1.22 or later
 - `scipy <http://www.scipy.org>`_
 - `astropy <http://www.astropy.org>`_
 - `PyYaml <http://http://pyyaml.org/>`_
 - `matplotlib <https://matplotlib.org/>`_
 - `scikit-image <https://scikit-image.org/>`_
 - `lmfit <https://lmfit.github.io/lmfit-py/>`_
 - `reproject <https://reproject.readthedocs.io/en/stable/>`_
 - `python-dateutil <https://pypi.org/project/python-dateutil>`_


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

Our recommended option is to perform isolated installations
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
to translate Cython code into C/C++.

The development version can can be checked out with::

    git clone https://github.com/guaix-ucm/numina.git

Building and installing
++++++++++++++++++++++++

To build and install numina, run::

    pip install .

You can all install the package in "editable" mode, including the "-e" option::

    pip install -e .

  
.. _virtualenv: https://virtualenv.pypa.io/
.. _sphinx: http://sphinx.pocoo.org
.. _pytest: http://pytest.org/latest/
.. _virtualenv_install: https://virtualenv.pypa.io/en/latest/installation.html
.. _virtualenv_usage: https://virtualenv.pypa.io/en/latest/userguide.html
