======
Numina
======

This is Numina, the data reduction package used by the following GTC
instruments: EMIR, FRIDA, MEGARA and MIRADAS.

Numina is distributed under GNU GPL, either version 3 of the License, 
or (at your option) any later version. See the file LICENSE.txt for 
details.

Requirements
------------

Python 2.7 is required. Numina requires the following 
packages installed in order to work properly:

 - numpy (http://numpy.scipy.org/) 
 - scipy (http://www.scipy.org)
 - pyfits (http://www.stsci.edu/resources/software_hardware/pyfits)
 - PyYaml (http://http://pyyaml.org/)
 
The documentation of the project is generated using Sphinx (http://sphinx.pocoo.org/)

Additional packages are optionally required:
 - sphinx (http://sphinx.pocoo.org) to build the documentation

Webpage: https://guaix.fis.ucm.es/projects/emir
Maintainer: sergiopr@fis.ucm.es


Stable version
--------------

The latest stable version of Numina can be downloaded from  
ftp://astrax.fis.ucm.es/pub/software/numina/

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

    $ hg clone https://guaix.fis.ucm.es/hg/numina

And then installed following the standard procedure:::

    $ cd numina
    $ python setup.py install

Building the documentation
---------------------------
The Numina documentation is base on `sphinx`_. With the package installed, the 
html documentation can be built from the `doc` directory::

  $ cd doc
  $ make html
  
The documentation will be copied to a directory under `build/sphinx`. 
  
The documentation can be built in different formats. The complete list will appear
if you type `make` 
  
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _sphinx: http://sphinx.pocoo.org

