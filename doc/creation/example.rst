
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
a dedicated subpackge for recipes `clodiadrp.recipes` and a module for each recipe. The file layout is::

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


Recipes must provide three things: a description of the inputs of the recipe, a description of the products of the recipe and a method
execute the processing. All Recipes must inherit from :class:`~numina.core.recipes.BaseRecipe`, so recipes will be classes.

We start with a simple `Bias` recipe. Its purpose is to process images taken in *Bias* mode, that is, a series of pedestal images.
The recipe will receive the result of the observation and return a master bias image.

.. code-block:: python

    from numina.core import Product, Requirement
    from numina.core import DataFrameType
    from numina.core.products import ObservationResultType
    from numina.core.recipes import BaseRecipe

    class Bias(BaseRecipe):                            (1)

        obresult = Requirement(ObservationResultType)  (2)
        master_bias = Product(DataFrameType)           (3)

        def run(self, rinput):                         (4)

            # Here the raw images are processed
            # and a final image myframe is created

            result = self.create_result(master_bias=myframe)  (5)
            return result

1. Each recipe must be a class derived from :class:`~numina.core.recipes.BaseRecipe`
2. This recipe only requires the result of the observation. Each requirement is an object of
   class :class:`~numina.core.requirements.Requirement` or its specialized subclasses. The actual
   type of the requirement is :class:`~numina.core.products.ObservationResultType`, representing
   the result of the observation.
3. This recipe only produces one result. Each product is an object of
   class :class:`~numina.core.dataholders.Product`. The actual type of the product is given by
   :class:`~numina.core.products.DataFrameType`, representing an image.
4. Each recipe must provide a `run` method. The method has only one argument that collects
   the values of all inputs declared by the recipe. In this case, `rinput` will have a member
   `rinput.obresult` of type :class:`~numina.core.oresult.ObservationResult`

5. The recipe must return an object that collects all the declared products of the recipe, of type
   :class:`~numina.core.recipeinout.RecipeResult`. This is acomplished internally by the method `create_result`
   It will raise an exception at run time if all the declared products are not provided.


We can create now the `Flat` recipe (inside `flat.py`). This recipe will have two requirements, the corresponding
observation result and a master bias image, as flat-field images require bias subtraction.

.. code-block:: python

    from numina.core import Product, Requirement
    from numina.core import DataFrameType
    from numina.core.products import ObservationResultType
    from numina.core.recipes import BaseRecipe

    class Flat(BaseRecipe):

        obresult = Requirement(ObservationResultType)  (1)
        master_bias = Requirement(DataFrameType)       (2)
        master_flat = Product(DataFrameType)

        def run(self, rinput):                          (3)

            # Here the raw images are processed
            # and a final image myframe is created

            result = self.create_result(master_flat=myframe)  (4)
            return result


1. This recipe still requires the result of the observation. Each requirement is an object of
   class :class:`~numina.core.requirements.Requirement` or its specialized subclasses. The actual
   type of the requirement is :class:`~numina.core.products.ObservationResultType`, representing
   the result of the observation.
2. And it also requires a master bias image, the type in this case is :class:`~numina.core.products.DataFrameType`,
   representing an image.
3. In this case, `rinput` will have two members
   `rinput.obresult` of type :class:`~numina.core.oresult.ObservationResult` and
   `rinput.master_bias` of type :class:`~numina.core.dataframe.DataFrame`
4. The arguments of `create_result` must be passed with the same name used in the product definition.

And finally, the recipe for reduction of `Image` mode  (inside `image.py`). This recipe will have three requirements, the corresponding
observation result and a master bias and master flat images

.. code-block:: python

    from numina.core import Product, Requirement
    from numina.core import DataFrameType
    from numina.core.products import ObservationResultType
    from numina.core.recipes import BaseRecipe

    class Image(BaseRecipe):

        obresult = Requirement(ObservationResultType)
        master_bias = Requirement(DataFrameType)
        master_flat = Requirement(DataFrameType)
        final = Product(DataFrameType)

        def run(self, rinput):                          (1)

            # Here the raw images are processed
            # and a final image myframe is created

            result = self.create_result(final=myframe)
            return result



1. In this case, `rinput` will have three members
   `rinput.obresult` of type :class:`~numina.core.oresult.ObservationResult`,
   `rinput.master_bias` of type :class:`~numina.core.dataframe.DataFrame` and
   `rinput.master_flat` of type :class:`~numina.core.dataframe.DataFrame`

.. note::

   It is not strictly required that the requirements and products names are
   consistent between recipes, altough is highly recomended

Now we must update `drp.yaml` to insert the full name of the recipes (package and class), as follows

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
          bias: clodiadrp.recipes.bias.Bias
          flat: clodiadrp.recipes.flat.Flat
          image: clodiadrp.recipes.image.Image


Specialized data products
*************************

There is some information that is missing of our current setup. The products of some recipes are the inputs of others.
The master bias created by `Bias` is the input that `Flat` and `Image` require. To represent this we use specialized
data products. We start by adding a new module `products`::

    clodiadrp
    |-- clodiadrp
    |   |-- __init__.py
    |   |-- loader.py
    |   |-- products.py
    |   |-- drp.yaml
    |   |-- recipes
    |   |   |-- __init__.py
    |   |   |-- bias.py
    |   |   |-- flat.py
    |   |   |-- image.py
    |-- setup.py

We have two types of images that are products of recipes that can be required by other recipes: **master bias**
and **master flat**. We represent this by creating two new types derived
from :class:`~numina.core.products.DataFrameType`, becasue the new types are images
and :class:`~numina.core.products.DataProductTag`, because the new types are products that must be handled specially
by Numina CLI and GTC Control system.

.. code-block:: python

    from numina.core.products import DataFrameType, DataProductTag

    class MasterBias(DataFrameType, DataProductTag):
        pass


    class MasterFlat(DataFrameType, DataProductTag):
        pass

Now we must modify our recipes as follows. First `Bias`

.. code-block:: python

    from numina.core import Product, Requirement
    from numina.core.products import ObservationResultType
    from numina.core.recipes import BaseRecipe
    from clodiadrp.products import MasterBias   (1)

    class Bias(BaseRecipe):

        obresult = Requirement(ObservationResultType)
        master_bias = Product(MasterBias)        (2)

        ...                                       (3)

1. We import the new type `MasterBias`
2. And declare that our recipe produces `MasterBias` images.
3. `run` remains unchanged.

Then `Flat`:

.. code-block:: python

    from numina.core import Product, Requirement
    from numina.core.products import ObservationResultType
    from numina.core.recipes import BaseRecipe
    from clodiadrp.products import MasterBias, MasterFlat

    class Flat(BaseRecipe):

        obresult = Requirement(ObservationResultType)
        master_bias = Requirement(MasterBias)         (1)
        master_flat = Product(MasterFlat)             (2)

        ...

1. We use now `MasterBias` as a requirement. This guaranties that the images provided
   here are those created by `Bias` and no other.
2. We declare that our recipe produces `MasterFlat` images.

And finally `Image`:


.. code-block:: python

    from numina.core import Product, Requirement
    from numina.core import DataFrameType
    from numina.core.products import ObservationResultType
    from numina.core.recipes import BaseRecipe
    from clodiadrp.products import MasterBias, MasterFlat

    class Image(BaseRecipe):

        obresult = Requirement(ObservationResultType)
        master_bias = Requirement(MasterBias)
        master_flat = Requirement(MasterFlat)
        final = Product(DataFrameType)

        ...
