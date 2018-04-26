***************************
Numina Pipeline Environment
***************************

This guide is intended as an introductory overview of pipeline creation
with Numina. For detailed reference documentation of the functions and
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


Recipe Example
##############
In order to make the new Pipeline visible to Numina you have to:

1.  Create a configuration yaml file.
2.  Create a loader file.
3.  Link the *entry_point* option in the *setup.py* with the loader file.
4.  Create the Pipeline's Recipes.

In the following we will continue with the same example as previously.


Configuration File
******************
Create a new yaml file in the root folder named *drp.yaml* with the following
information:

.. code-block:: yaml

    name: CLODIA
    configurations:
      default: {}
    modes:
    - date: 2015-11-12
      description: Description of CLODIA Instrument
      key: clodia_key
      name: Clodia_Name
      reference: numina
      status: FINAL
      summary: Summary of CLODIA Instrument
      tagger: null
    pipelines:
      default:
        recipes:
          clodia_key: clodiadrp.recipes.name_of_recipe
        version: 1
    #products:
    #- name: clodiadrp.products.TraceMap
    #  alias: TraceMap


Loader File
***********
Create a new loader file in the root folder named *loader.py* with the
following information:

.. code-block:: python

    from numina.core import drp_load

    def drp_load():
        '''Entry point to load CLODIA DRP.'''
        return drp_load('clodiadrp', 'drp.yaml')


Link Files
**********
Once we have created the *loader.py* file, the only thing we have to do to
make CLODIA visible to Numina is to modify the *setup.py* file

.. code-block:: python

    from setuptools import setup

    setup(name='clodiadrp',
          entry_points = {
            'numina.pipeline.1': ['CLODIA = clodiadrp.loader:drp_load'],
            },
    )


Recipes Creation
****************
All new Recipes inherit from **BaseRecipe** class that can be found in
*numina.core.recipes* so following with the example, we want to create a new
recipe in the folder *recipes* of clodiadrp: *clodiadrp.recipes.name_of_recipe*
with the following code:

.. code-block:: python

    from numina.core import Result
    from numina.core.recipes import BaseRecipe
    from numina.core.requirements import ObservationResultRequirement
    import numpy as np
    from astropy.io import fits

    class name_of_recipe(BaseRecipe):

        obresult = ObservationResultRequirement()
        output_image = Result()

        def run(self, rinput):

            n = np.arange(50.0)
            hdu = fits.PrimaryHDU(n)

            result = self.create_result(output_image=hdu)
            return result

Directory structure
*******************

At the end, our files in the example should follow the next directory
structure::

    clodiadrp
    |-- recipes
    |   |-- __init__.py
    |   |-- name_of_recipe.py
    |-- products
    |   |-- __init__.py
    |-- setup.py
    |-- loader.py
    |-- drp.yam


Note that it is quite important to specify all required arguments (obresult)
and the output (output_image). When the reduction is run from the command line
using Numina CLI, it checks that the required values are provided or have
default values.

Furthermore, if the Recipe is properly configured it is executed with an
observing block data structure as input. Otherwise, when it is run using
Numina CLI, the data structure is created from a text file.




