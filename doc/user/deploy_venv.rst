.. _deploy_venv:

=================================
Numina Deployment with Virtualenv
=================================

`Virtualenv`_ is a tool to build isolated Python environments.

Since Python version 3.3, there is also a module in the standard library
called `venv` with roughly the same functionality.


Create Virtual Environment
--------------------------

In order to create a virtual environment called e.g. numinaenv using venv::

    python3 -m venv numinaenv


Activate the Environment
------------------------
Once the environment is created, you need to activate it. The activation script in
in the `bin/` folder created under `numinaenv. Just source the
script `activate`::

  source numinaenv/bin/activate
  (numinaenv) $

Notice that the prompt changes once you have activated the environment. To
deactivate it, just type `deactivate`::

  (numinaenv) $ deactivate
  $ 

.. note:: We are assuming that the user shell is bash. There are alternative *activate*
            scripts for tcsh and fish called `activate.csh` and `activate.fish`


Numina Installation
-------------------
After the environment activation, we can install numina with pip.
This is the standard Python tool for package management. It will download the package and its
dependencies, unpack everything and compile when needed::

  (numinaenv) $ pip install numina
  
The requirements of numina will be downloaded and installed inside
the virtual environment automatically.

You can also update numina, if your environment contains already a installed version::

    (numinaenv) $ pip install -U numina

You can also install directly from the `git` repository. `pip` can access individual
branches, particular commits or just the latest code. For example this will install
the latest development version::

    (numinaenv) $ pip install git+https://github.com/guaix-ucm/numina.git

Check the `pip` documentation for the syntax of other types of packages sources.

Test the installation
---------------------

We can test the installation by running the ``numina`` command:

::

    (numinaenv) $ numina
    INFO: Numina simple recipe runner version 0.20


.. _virtualenv: https://virtualenv.pypa.io/

