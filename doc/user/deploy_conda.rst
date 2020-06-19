.. _deploy_conda:

============================
Numina Deployment with Conda
============================

`Conda <https://conda.io/docs/>`_ was created with a target similar to
`virtualenv`_, but extended its functionality to the management of packages in
different languages.

You can install `miniconda <https://conda.io/miniconda.html>`_ or `anaconda
<http://docs.anaconda.com/anaconda/install/>`_. The difference is that
miniconda provides a light-weight environment and anaconda comes with lots of
additional Python packages. By installing ``miniconda`` you reduce the amount
of preinstalled packages in your system (after installing ``miniconda`` it is
possible to install ``anaconda`` by executing ``conda install anaconda``).

If you have updated the ``$PATH`` system variable during the miniconda or conda
installation, you can call conda commands directly in the shell, like this:

::

   bash$ conda info

If not, you will need the add the path to the command, like:

::

  bash$ /path/to/conda/bin/conda info


In this guide we will write the commands without the full path, for simplicity.

Once conda is installed according to the corresponding miniconda or anaconda
instructions, the steps to instal numina under conda are:

Create a conda environment
--------------------------

With coda, environments are created in a centralised manner (under the
subdirectory ``./envs`` in your conda tree)::

    conda create --name numinaenv

The Pyhton interpreter used in this environment is the same version
currently used by conda. You can select a different version with::

    conda create --name numinaenv python=3.6


Activate the environment
------------------------

With command::


    conda activate numinaenv

which yields a different system prompt to the user::

     (numinaenv) $


To exit the environment is enough to exit the terminal or run the
following command::

     (numinaenv) $ conda deactivate


Numina installation
-------------------
After the environment activation, we can install numina using conda (we
provide conda packages in the `conda-forge <https://conda-forge.org>`_
channel)::

     (numinaenv) $ conda install -c conda-forge numina

We can also update numina, if your environment contains already a installed version::

    (numinaenv) $ conda update numina

If you need to install the development version, you can download the source code and
proceed following the instructions in xxxx.

Other possibility is using `pip`. It can access individual
branches, particular commits or just the latest code. For example this will install
the latest development version::

    (numinaenv) $ pip install git+https://github.com/guaix-ucm/numina.git

Check the `pip` documentation for the syntax of other types of packages sources.

.. warning:: Issues can arise with packages installed with `pip` in `conda` environments.
              See https://www.anaconda.com/blog/using-pip-in-a-conda-environment for details.


Test the installation
---------------------

We can test the installation by running the ``numina`` command:

::

    (numinaenv) $ numina
    INFO: Numina simple recipe runner version 0.20


.. _virtualenv: https://virtualenv.pypa.io/