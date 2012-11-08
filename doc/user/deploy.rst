
=================================
Numina Deployment with Virtualenv
=================================

`Virtualenv`_ is a tool to build isolated Python environments.

It's a great way to quickly test new libraries without cluttering your 
global site-packages or run multiple projects on the same machine which 
depend on a particular library but not the same version of the library.

Install virtualenv
------------------
I install it with the package system of my OS, so that it ends in my
global site-packages.

With Fedora/EL is just::

  $ sudo yum install python-virtualenv

For other ways of installing the package, check `virtualenv`_ webpage.


Create virtual environment
--------------------------
Create the virtual environment enabling the packages already installed
in the global site-packages via the OS package system. Some requirements
(in particullar numpy and scipy) are difficult to build: they require
compiling and external C and FORTRAN libraries to be installed. Hopefully
you have already installed numpy and scipy.

So the command is::

  $ virtualenv --system-site-packages myenv

If you need to create the virtualenv without global packages, drop the
system-site-packages flag and add :option:`--no-site-packages`.

Activate the environment
-------------------------
Once the environment is created, you need to activate it. Just change
directory into it and load with your command line interpreter the 
script bin/activate.

With bash::

  $ cd myenv
  $ . bin/activate
  (myenv) $

With csh/tcsh::

  $ cd myenv
  $ source bin/activate.csh
  (myenv) $

Notice that the prompt changes once you are activate the environment. To 
deactivate it just type deactivate::

  (myenv) $ deactivate
  $ 

Install Numina
---------------

Numina is registered in the Python Package Index. That means (among 
other things) that can be installed inside the environment with one command.::

  (myenv) $ pip install numina
  
The requirements of numina will be downloaded and installed inside
the virtual environment.

.. _virtualenv: http://pypi.python.org/pypi/virtualenv
