============
Installation
============

Requirements
------------

Python 2.7 or Python >= 3.4 is required. Additionally the following packages are required
in order to work properly:

 - `setuptools <http://pythonhosted.org/setuptools/>`_
 - `six <http://pythonhosted.org/six/>`_
 - `numpy <http://numpy.scipy.org/>`_ 
 - `scipy <http://www.scipy.org>`_
 - `astropy <http://www.astropy.org>`_
 - `PyYaml <http://http://pyyaml.org/>`_
 - `dateutil <https://pypi.org/project/python-dateutil>`_

For Pyhton 2.7, the following compatibility packages are required:
 - `singledispatch <https://pypi.python.org/pypi/singledispatch>`_
 - `enum34 <https://pypi.org/project/enum34/>`_

The following packages are optional:
 - `sphinx`_  to build the documentation
 - `pytest`_  for testing

Stable version
--------------

The latest stable version of Numina can be downloaded from  
https://pypi.python.org/pypi/numina/

To install Numina, use the standard installation procedure::

    $ tar zxvf numina-X.Y.Z.tar.gz
    $ cd numina-X.Y.Z
    $ python setup.py install
    
The `install` command provides options to change the target directory. By default
installation requires administrative privileges. The different installation options
can be checked with:: 

   $ python setup.py install --help
   
Development version
-------------------

The development version can be checked out with::

    $ git clone https://github.com/guaix-ucm/numina.git

And then installed following the standard procedure::

    $ cd numina
    $ python setup.py install

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
