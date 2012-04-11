======
Numina
======

This is Numina, the data reduction package used by the following GTC
instruments: EMIR, FRIDA, MEGARA, MIRADAS

Numina is distributed under GNU GPL, either version 3 of the License, 
or (at your option) any later version. See the file COPYING for details.

Numina requires the following packages installed in order to
work properly:

 - numpy (http://numpy.scipy.org/) 
 - scipy (http://www.scipy.org)
 - pyfits (http://www.stsci.edu/resources/software_hardware/pyfits)
 - PyYaml (http://http://pyyaml.org/)
 
The documentation of the project is generated using Sphinx (http://sphinx.pocoo.org/)
 

Webpage: https://guaix.fis.ucm.es/projects/emir
Maintainer: sergiopr@fis.ucm.es


Compilation 
-----------

Run the following command:

   python setup.py build


For additional options:

   python setup.py build --help


Installation
------------

   python setup.py install

For additional options:

   python setup.py install --help
