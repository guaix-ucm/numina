===================
Numina Installation
===================

This is Numina, the data reduction package used by the following GTC
instruments: EMIR, FRIDA, MEGARA and MIRADAS.

Numina is distributed under GNU GPL, either version 3 of the License, 
or (at your option) any later version. See the file LICENSE.txt 
for details.

Requirements
------------

Python >= 2.7 is required. Additionally the following packages are required
in order to work properly:

 - `setuptools <http://pythonhosted.org/setuptools/>`_
 - `six <http://pythonhosted.org/six/>`_
 - `numpy <http://numpy.scipy.org/>`_ 
 - `scipy <http://www.scipy.org>`_
 - `astropy <http://www.astropy.org>`_
 - `PyYaml <http://http://pyyaml.org/>`_
 - `singledispatch <https://pypi.python.org/pypi/singledispatch>`_
 (only if Python < 3.4)

Additional packages are optionally required:
 - `sphinx`_  to build the documentation
 - `pytest`_  for testing

Webpage: https://guaix.fis.ucm.es/projects/numina

Maintainer: sergiopr@fis.ucm.es

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
