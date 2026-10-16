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
instructions, the steps to install numina under conda are:

Create a conda environment
--------------------------

With conda, environments are created in a centralised manner (under the
subdirectory ``./envs`` in your conda tree)::

    conda create --name numinaenv

The Python interpreter used in this environment is the same version
currently used by conda. You can select a different version (3.11 or
later) with::

    conda create --name numinaenv python=3.12


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
After the environment activation, we install numina with ``pip``, inside the
conda environment. pip downloads numina and its dependencies from `PyPI
<https://pypi.org/project/numina/>`_::

     (numinaenv) $ pip install numina

We can also update numina, if your environment contains already a installed version::

    (numinaenv) $ pip install -U numina

.. note:: The packages of numina in the `conda-forge <https://conda-forge.org>`_
          channel are no longer updated: they are older versions, and some
          dependencies of numina, such as teareduce, are not in conda-forge.

pip can also install individual branches, particular commits or just the
latest code. For example this will install the latest development version::

    (numinaenv) $ pip install git+https://github.com/guaix-ucm/numina.git

Check the `pip` documentation for the syntax of other types of packages sources,
and :ref:`deploy_source` to build numina from the source code.

.. warning:: Install numina and its dependencies with pip, and avoid installing
              or updating the same packages afterwards with conda in that
              environment. See https://www.anaconda.com/blog/using-pip-in-a-conda-environment
              for details.


Test the installation
---------------------

We can test the installation by running the ``numina`` command:

::

    (numinaenv) $ numina --version
    numina 0.40.0


.. _virtualenv: https://virtualenv.pypa.io/
