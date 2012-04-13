
.. _user:

#################
Numina User Guide
#################

This guide is intended as an introductory overview of Numina and
explains how to install and make use of the most important features of
Numina. For detailed reference documentation of the functions and
classes contained in the package, see the :ref:`reference`.

.. warning::

   This "User Guide" is still a work in progress; some of the material
   is not organized, and several aspects of Numina are not yet covered
   sufficient detail.
   
===================
Numina Installation
===================

This is Numina, the data reduction package used by the following GTC
instruments: EMIR, FRIDA, MEGARA and MIRADAS

Numina is distributed under GNU GPL, either version 3 of the License, 
or (at your option) any later version. See the file COPYING for details.

Numina requires the following packages installed in order to
be able to be installed and work properly:

 - pyfits (http://www.stsci.edu/resources/software_hardware/pyfits)
 - setuptools (http://peak.telecommunity.com/DevCenter/setuptools)
 - numpy (http://numpy.scipy.org/)
 - scipy (http://www.scipy.org)
 - pyxdg (http://www.freedesktop.org/wiki/Software/pyxdg)
 - simplejson (http://undefined.org/python/#simplejson)

Additional packages are optionally required:
 - nose (http://somethingaboutorange.com/mrl/projects/nose) to run the tests
 - sphinx (http://sphinx.pocoo.org) to build the documentation

Webpage: https://guaix.fis.ucm.es/projects/emir

Maintainer: sergiopr@fis.ucm.es


Building 
--------

Run the following command::

   $ python setup.py build


For additional options::

   $ python setup.py build --help

Building the documentation
---------------------------
The Numina documentation is base on sphinx. With the package installed, the 
documentation can be built with a custom buld target::

  $ python setup.py build_sphinx


Installation
------------

Run the command::

   $ python setup.py install

For additional options::

   $ python setup.py install --help
