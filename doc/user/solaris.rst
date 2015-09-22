
.. _solaris10:

===============================
Numina Deployment in Solaris 10
===============================

Solaris 10 is the Operative System (OS) under a substantial part
of the GTC Control System runs. The installation of the
Python stack in this OS is not trivial, so in the following a description
of the required steps is done.


Basic Tools Installation
------------------------
Firstly a GNU compiler collection should be installed (compilers for C,
C++ and Fortran). The `opencsw`_ project
provides precompiled binaries of these programs.
Refer to the
`project's documentation <http://www.opencsw.org/manual/for-administrators/getting-started.html#getting-started>`_
to setup opencsw in the system and then install with:

::

  /opt/csw/bin/pkgutil -i CSWgcc4core
  /opt/csw/bin/pkgutil -i CSWgcc4g++
  /opt/csw/bin/pkgutil -i CSWgcc4gfortran

Additionally, both the Pyhton program and the developer tools can also be installed from opencsw

::

  /opt/csw/bin/pkgutil -i CSWpython27
  /opt/csw/bin/pkgutil -i CSWpython27-dev


ATLAS and LAPACK Installation
-----------------------------
`ATLAS`_ is a linear algebra library.
Numpy can be installed without any linear algebra library, but scipy can't.

`LAPACK`_ provides Fortran routines for solving
systems of simultaneous linear equations, least-squares solutions of linear
systems of equations, eigenvalue problems, and singular value problems.

ATLAS need to be built with LAPACK support, so both libraries can be found at
`source code of ATLAS
<http://sourceforge.net/projects/math-atlas/files/Stable/>`_
and
`source code of LAPACK
<http://www.netlib.org/lapack/#_previous_release>`_.

Once the source code of ATLAS and LAPACK are downloaded, the instructions
to build them can be found at
`build documentation <http://math-atlas.sourceforge.net/atlas_install/>`_
which basically requires to setup a different directory to run
the ``configure`` command in it and then ``make install``.

As an example, these ``configure`` and ``make`` lines are used in
our development machine:

::

  ../configure --cc=/opt/csw/bin/gcc --shared --with-netlib-lapack-tarfile=/path/to/lapack-3.5.0.tar.gz --prefix=/opt/atlas
  make
  make install

The install step may require root privileges. The libraries and headers will
be installed under some prefix (in our case, ``/opt/atlas/include`` and
``/opt/atlas/lib``).

Numpy Installation
------------------
Download the latest numpy source code from `numpy's webpage <http://www.scipy.org/install.html#individual-binary-and-source-packages>`_.

Numpy source distribution contains a file called ``site.cfg``
which describes the different types of linear algebra libraries present in
the system.
Copy ``site.cfg.example`` to ``site.cfg`` and edit
the section containing the ATLAS libraries. Everything in the file should
be commented except the following

::

  [atlas]
  library_dirs = /opt/atlas/lib
  include_dirs = /opt/atlas/include

The paths should point to the version of ATLAS installed in the system.

Other packages (such as scipy) will also use a ``site.cfg`` file. To avoid
editing the same file again, copy ``site.cfg`` to ``.numpy-site.cfg`` in
the ``$HOME`` directory.

::

 cp site.cfg $HOME/.numpy-site.cfg

After this configuration step, numpy should be built.

::

  python setup.py build
  python setup.py install --prefix /path/to/my/python/packages

The last step may require root privileges. Notice that you can use
``--user`` instead of ``--prefix`` for local packages.


Scipy Installation
------------------
As of this writing, the last released version of scipy is 0.15.1 and it
doesn't work in Solaris 10 `due to a bug <https://github.com/scipy/scipy/issues/4704>`_  [1]_.

This bug may be fixed in next stable release
(check the release notes of scipy), but meanwhile a patch can be used.

Download the scipy 0.15.1 source code from `scipy's webpage <http://scipy.org/install.html#individual-binary-and-source-packages>`_.  Then download the patch: `scipy151-solaris10.patch <https://guaix.fis.ucm.es/~spr/scipy151-solaris10.patch>`_.

Extract the source code and apply the patch with the command:

::

 patch -p1 -u -d scipy-0.15.1 < scipy151-solaris10.patch

After this step, build and install scipy normally.

::

  python setup.py build
  python setup.py install --prefix /path/to/my/python/packages

During the build step, local ``.numpy-site.cfg`` will be read so the
path to the ATLAS libraries will be used.

The prefix used to install scipy must be the same than the used with numpy.
In general all python packages must be installed under the same prefix.


Pip Installation
----------------

To install pip, download `get-pip.py
<https://bootstrap.pypa.io/get-pip.py>`_.

Then run the following:

::

 python get-pip.py

Refer to https://pip.pypa.io/en/latest/installing.html#install-pip
to more detailed documentation.

Numina Installation
-------------------
Finally, numina can be installed directly using ``pip``. Remember to set
the same prefix used previously with numpy and scipy.

::

  pip install numina --prefix /path/to/my/python/packages


----

.. [1] https://github.com/scipy/scipy/issues/4704

.. _ATLAS:  http://math-atlas.sourceforge.net/
.. _LAPACK: http://www.netlib.org/lapack/
.. _opencsw: http://www.opencsw.org/
