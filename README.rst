======
Numina
======

|zenodo| |docs| 

.. image:: https://travis-ci.org/guaix-ucm/numina.svg?branch=master
    :target: https://travis-ci.org/guaix-ucm/numina

.. image:: https://coveralls.io/repos/guaix-ucm/numina/badge.svg?branch=master&service=github 
    :target: https://coveralls.io/github/guaix-ucm/numina?branch=master 

.. image:: https://readthedocs.org/projects/numina/badge/?version=latest
    :target: http://numina.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

This is Numina, the data reduction package used by the following GTC
instruments: EMIR, FRIDA, MEGARA and MIRADAS.

Numina is distributed under GNU GPL, either version 3 of the License, 
or (at your option) any later version. See the file LICENSE.txt for 
details.

Requirements
------------

Python 2.7 or 3.3 (or greater) is required. Numina requires the following 
packages installed in order to work properly:

 - setuptools (http://pythonhosted.org/setuptools/)
 - six >= 1.7 (https://pythonhosted.org/six/)
 - numpy >= 1.6 (http://numpy.scipy.org/) 
 - scipy (http://www.scipy.org)
 - astropy >= 1.0 (http://www.astropy.org/)
 - PyYaml (http://pyyaml.org/)
 - singledispatch (only if Python < 3.4, https://pypi.python.org/pypi/singledispatch)
 
The documentation of the project is generated using Sphinx (http://sphinx.pocoo.org/)

Additional packages are optionally required:
 - sphinx (http://sphinx-doc.org) to build the documentation
 - pytest (http://pytest.org) for testing

Webpage: https://guaix.fis.ucm.es/projects/numina
Maintainer: sergiopr@fis.ucm.es


Stable version
--------------

The latest stable version of Numina can be downloaded from  
https://pypi.python.org/pypi/numina

To install numina, use the standard installation procedure:::

    $ tar zxvf numina-X.Y.Z.tar.gz
    $ cd numina-X.Y.Z
    $ python setup.py install
    

The `install` command provides options to change the target directory. By default
installation requires administrative privileges. The different installation options
can be checked with::: 

   $ python setup.py install --help
   
Development version
-------------------

The development version can be checked out with:::

    $ git clone https://github.com/guaix-ucm/numina.git

And then installed following the standard procedure:::

    $ cd numina
    $ python setup.py install

Building the documentation
---------------------------
The Numina documentation is based on `sphinx`_. With the package installed,
the html documentation can be built from the `doc` directory::

  $ cd doc
  $ make html
  
The documentation will be copied to a directory under `build/sphinx`. 
  
The documentation can be built in different formats. The complete list will appear
if you type `make` 
  
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _sphinx: http://sphinx.pocoo.org

.. |docs| image:: https://readthedocs.org/projects/numina/badge/?version=latest
    :alt: Documentation Status
    :target: https://readthedocs.org/projects/numina/?badge=latest

.. |zenodo| image:: https://zenodo.org/badge/13088/guaix-ucm/numina.svg
   :target: http://dx.doi.org/10.5281/zenodo.17986
