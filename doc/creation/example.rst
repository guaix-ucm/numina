
***********************
Numina Pipeline Example
***********************

This guide is intended as an introductory overview of the creation of
instrument reduction pipelines with Numina. For detailed reference
documentation of the functions and
classes contained in the package, see the :ref:`reference`.

.. warning::

   This "Pipeline Creation Guide" is still a work in progress; some of 
   the material
   is not organized, and several aspects of Numina are not yet covered
   sufficient detail.

Execution environment of the Recipes
####################################

Recipes have different execution environments. Some recipes are designed
to process observing modes required for the observation. These modes
are related to visualization, acquisition and focusing. The Recipes
are integrated in the GTC environment. We call these recipes the
**Data Factory Pipeline**, (:term:`DFP`).

Other group of recipes are devoted to scientific observing modes: imaging, 
spectroscopy and auxiliary calibrations. These Recipes constitute the
**Data Reduction Pipeline**, (:term:`DRP`). The software is meant to be standalone,
users shall download the software and run it in their own computers, with
reduction parameters and calibrations provided by the instrument team.

Users of the DRP will use the simple Numina CLI.
Users of the DFP shall interact with the software
through the GTC Inspector.


Instrument Reduction Pipeline Example
#####################################

In the following sections we create an Instrument Reduction Pipeline for an instrument
name *CLODIA*.

In order to make a new Instrument Reduction Pipeline
visible to Numina and the GTC Control System you have to create a full Python package
that will contain the reduction recipes, data types and other processing code.

The creation of Python packages is described in detail (for example) in the
`Python Packaging User Guide <http://python-packaging-user-guide.readthedocs.org>`_.

Then, we create a Python package called `clodiadrp` with the following structure (we
ignore files such as README or LICENSE as they are not relevant here)::

    clodiadrp
    |-- clodiadrp
    |   |-- __init__.py
    |-- setup.py

From here the steps are:

1.  Create a configuration yaml file.
2.  Create a loader file.
3.  Link the *entry_point* option in the *setup.py* with the loader file.
4.  Create the Pipeline's Recipes.

In the following we will continue with the same example as previously.


Configuration File
******************

The configuration file contains basic informatio such as

  * the list of modes of the instrument
  * the list of recipes of the instrument
  * the mapping between recipes and modes.

In this example, we assume that CLODIA has three modes: **Bias**, **Flat** and **Image**.
The first two modes are used for pedestal and flat-field illumination correction. The third
is the main scientific mode of the instrument.

Create a new yaml file in the root folder named *drp.yaml*.

.. code-block:: yaml

    name: CLODIA
    configurations:
      default: {}
    modes:
     -key: bias
      name: Bias
      summary: Bias mode
      description: >
        Full description of the Bias mode
     -key: flat
      name: Flat
      summary: Flat mode
      description: >
        Full description of the Flat mode
     -key: image
      name: Image
      summary: Image mode
      description: >
        Full description of the Image mode
    pipelines:
      default:
        version: 1
        recipes:
          bias: clodiadrp.recipes.recipe
          flat: clodiadrp.recipes.recipe
          image: clodiadrp.recipes.recipe

The entry `modes` contains a list of the observing modes of the instrument. There are three: Bias, Flat and Image.
Each entry contains information about the mode. A *name*, a short *summary* and a multi-line *description*.
The field *key* is used to map the observing modes to the recipes under *recipes* below, so *key*
has to be unique and equal to only one value in each `recipes` block under `pipelines`.

The entry `pipelines` contains only one pipeline, called *default* by convention. The `pipeline` contains
recipes, each related to one observing mode by means of the filed *key*. For the moment we haven't developed any recipe,
so the value of each key (*clodiadrp.recipes.recipe*) doesn't exist yet.

.. note::

    This file has to be included in `package_data` inside `setup.py` to be distributed
    with the package, see
    `Installing Package Data <https://docs.python.org/3/distutils/setupscript.html#installing-package-data>`_
    for details.

Loader File
***********
Create a new loader file in the root folder named *loader.py* with the
following information:

.. code-block:: python

    import numina.core

    def drp_load():
        """Entry point to load CLODIA DRP."""
        return numina.core.drp_load('clodiadrp', 'drp.yaml')


Create entry point
******************
Once we have created the *loader.py* file, the only thing we have to do to
make CLODIA visible to Numina/GCS is to modify the *setup.py* file to add an
entry point.

.. code-block:: python

    from setuptools import setup

    setup(name='clodiadrp',
          entry_points = {
            'numina.pipeline.1': ['CLODIA = clodiadrp.loader:drp_load'],
            },
    )

Both the Numina CLI tool and GCS check this particular entry point. They call the function provided
by the entry point. The function :func:`~numina.core.pipelineload.drp_load` reads and parses the YAML file and
creates an object of class :class:`~numina.core.pipeline.InstrumentDRP`. These objects are used by Numina CLI and GCS
to discover the available Instrument Reduction Pipelines.

At this stage, the file layout is as follows::

    clodiadrp
    |-- clodiadrp
    |   |-- __init__.py
    |   |-- loader.py
    |   |-- drp.yaml
    |-- setup.py


.. note::

    In fact, it is not necessary to use a YAML file to contain the Instrument information. The only
    strict requirement is that the function in the entry point 'numina.pipeline.1' must return
    a valid :class:`~numina.core.pipeline.InstrumentDRP` object. The use of a YAML file and the
    :func:`~numina.core.pipelineload.drp_load` function is only a matter of convenience.


Recipes Creation
****************
We haven't created any reduction recipe yet. As a matter of orgnization, we suggest to create
a dedicated subpackge porrecipes `clodiadrp.recipes` and a module for each recipe. The file layout is::

    clodiadrp
    |-- clodiadrp
    |   |-- __init__.py
    |   |-- loader.py
    |   |-- drp.yaml
    |   |-- recipes
    |   |   |-- __init__.py
    |   |   |-- bias.py
    |   |   |-- flat.py
    |   |   |-- image.py
    |-- setup.py


All new Recipes must inherit from :class:`~numina.core.recipes.BaseRecipe` so following with the example,
we want to create a new recipe in the folder *recipes* of clodiadrp: *clodiadrp.recipes.name_of_recipe*
with the following code:

.. code-block:: python

    from numina.core import Product, DataFrameType
    from numina.core.recipes import BaseRecipe
    import numpy as np
    from astropy.io import fits

    class name_of_recipe(BaseRecipe):

        output_image = Product(DataFrameType)

        def run(self, rinput):

            n = np.arange(50.0)
            hdu = fits.PrimaryHDU(n)

            result = self.create_result(output_image=hdu)
            return result

Directory structure
*******************

At the end, our files in the example should follow the next directory
structure::

    setup.py
    clodiadrp
    |-- recipes
    |   |-- __init__.py
    |   |-- name_of_recipe.py
    |-- products
    |   |-- __init__.py
    |-- loader.py
    |-- drp.yaml





