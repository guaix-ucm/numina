
=================================
Numina Deployment in Solaris 10
=================================

Solaris 10 is the operative system under a substantial part
of the GTC Control System runs. The installation of the
Python stack in this SO is not trivial, so we describe here
the required steps.


Install basic tools
-------------------
First we install the GNU compiler collection. We will need
compilers for C, C++ and Fortran. The `opencsw`_ project
provides precompiled binaries of these programs.
Refer to the `project documentation <http://www.opencsw.org/manual/for-administrators/getting-started.html#getting-started>`_ to setup opencsw in the system and then install with:

::

  /opt/csw/bin/pkgutil -i CSWgcc4core
  /opt/csw/bin/pkgutil -i CSWgcc4g++
  /opt/csw/bin/pkgutil -i CSWgcc4gfortran

The Pyhton program can be installed also from opencsw

::

  /opt/csw/bin/pkgutil -i CSWpython27
  /opt/csw/bin/pkgutil -i CSWpython27-dev


Install ATLAS and blas
--------------------------
`ATLAS <http://math-atlas.sourceforge.net/>`_ is a linear algebra library.
Numpy can be installed without any linear algebra library, but scipy can't.

We need to build ATLAS with `lapack`_ support, so we download the `source
code of lapack <http://www.netlib.org/lapack/#_previous_release>`_.


Once we have the source code of ATLAS and lapack, we follow the
`documentation <http://math-atlas.sourceforge.net/atlas_install/>`_
which bassically requires to setup a different directory to run
the ``configure`` command in it and then ``make``.

As an example, this configure line is used our development machine:

::

  ../configure --cc=/opt/csw/bin/gcc --shared --with-netlib-lapack-tarfile=/path/to/lapack-3.5.0.tar.gz --prefix=/opt/atlas
  make
  make install

The install step may require root privileges.

After following the instructions in ATLAS documentation, the libraries will be
installed under some prefix (in out case, ``/opt/atlas/include`` and
``/opt/atlas/lib``).

Install numpy
--------------
Download the latest numpy source code from `numpy webpage <http://www.scipy.org/install.html#individual-binary-and-source-packages>`_.

Numpy source distribution contains a file called ``site.cfg``
that describes the different
types of linear algebra libraries present in the system.
Copy ``site.cfg.example`` to ``site.cfg`` and edit
the section containing the atlas libraries. Everything in the file should
be commented except the following

::

  [atlas]
  library_dirs = /opt/atlas/lib
  include_dirs = /opt/atlas/include

The pate should paths should point to the version of ATLAS installed in the
system.

Other packages (such as scipy) will use also a ``site.cfg`` file. To avoid
editing the same file again, we can copy ``site.cfg`` to ``.numpy-site.cfg`` in
the ``$HOME`` directory.

::

 cp site.cfg $HOME/.numpy-site.cfg

After this configuration step, numpy should build.

::

  python setup.py build
  python setup.py install --prefix /path/to/my/python/packages

The last step may require root privileges. Notice that you can use
``--user`` instead of ``--prefix`` for local packages.


Install scipy
--------------
As of this writing, the last released version of scipy is 0.15.1 and it
doesn't work in Solaris 10 `due to a bug <https://github.com/scipy/scipy/issues/4704>`_  [1]_.

This bug may be fixed in next stable release
(check the release notes of scipy), but meanwhile a patch can be used.

Download the scipy 0.15.1 source code from `scipy webpage <http://scipy.org/install.html#individual-binary-and-source-packages>`_.  Then download the patch `scipy151-solaris10.patch <https://guaix.fis.ucm.es/~spr/scipy151-solaris10.patch>`_.

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
In general all the python packages must be installed under the same prefix.


Install pip
------------

To install pip, download `get-pip.py
<https://bootstrap.pypa.io/get-pip.py>`_.

Then run the following:

::

 python get-pip.py

Refer to https://pip.pypa.io/en/latest/installing.html#install-pip
to more detailed documentation.

Install numina
---------------
Finally, numina can be installed directly using ``pip``. Remember to set
the same prefix used previously with numpy and scipy.

::

  pip install numina --prefix /path/to/my/python/packages


----

.. [1] https://github.com/scipy/scipy/issues/4704

.. _atlas:  http://math-atlas.sourceforge.net/
.. _lapack: http://www.netlib.org/lapack/
.. _opencsw: http://www.opencsw.org/
