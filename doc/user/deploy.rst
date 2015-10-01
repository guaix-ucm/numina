
=================================
Numina Deployment with Virtualenv
=================================

`Virtualenv`_ is a tool to build isolated Python environments.

It's a great way to quickly test new libraries without cluttering your 
global site-packages or run multiple projects on the same machine which 
depend on a particular library but not the same version of the library.

Install Virtualenv
------------------
To install globally with pip (if you have pip 1.3 or greater installed globally)::

  $ sudo yum install python-virtualenv

For other ways of installing the package, check `virtualenv_install`_ webpage.


Create Virtual Environment
--------------------------
We urge reader to read the `virtualenv_usage`_ webpage to use and create
new virtual environments.

As an example, a new virtual environment named numina is created where no
packages but pip and setuptools are installed::

  $ virtualenv numina


Activate the Environment
------------------------
Once the environment is created, you need to activate it. Just go to `bin/` folder
created under numina and  load with your command line interpreter the
script bin/activate::

  $ cd numina/bin
  $ source activate
  (numina) $

Notice that the prompt changes once you are activate the environment. To 
deactivate it just type `deactivate`::

  (numina) $ deactivate
  $ 

Numina Installation
-------------------
Numina is registered in the Python Package Index. That means (among 
other things) that can be installed inside the environment with one command::

  (numina) $ pip install numina
  
The requirements of numina will be downloaded and installed inside
the virtual environment automatically.

.. _virtualenv: https://virtualenv.pypa.io/
.. _virtualenv_install: https://virtualenv.pypa.io/en/latest/installation.html
.. _virtualenv_usage: https://virtualenv.pypa.io/en/latest/userguide.html
