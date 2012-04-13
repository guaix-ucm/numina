
.. _creation:

##############################
Numina Pipeline Creation Guide
##############################

This guide is intended as an introductory overview of pipeline creation
with Numina. For detailed reference documentation of the functions and
classes contained in the package, see the :ref:`reference`.

.. warning::

   This "Pipeline Creation Guide" is still a work in progress; some of 
   the material
   is not organized, and several aspects of Numina are not yet covered
   sufficient detail.

Execution environment of the Recipes
------------------------------------

Recipes have different execution environments. Some recipes are designed
to process observing modes required for the observation. These modes
are related to visualization, acquisition and focusing. The Recipes
are integrated in the GTC environment. We call these recipes the
**Data Factory Pipeline**, (DFP).

Other group of recipes are devoted to scientific observing modes: imaging, 
spectroscopy and auxiliary calibrations. These Recipes constitute the
**Data Reduction Pipeline**, (DRP). The software is meant to be standalone,
users shall download the software and run it in their own computers, with
reduction parameters and calibrations provided by the instrument team.

Users of the DRP may use the simple Numina CLI or the higher level,
database-driven Pontifex. Users of the DFP shall interact with the software
through the GTC Inspector.

Recipe Parameters
-----------------
Recipes based on Numina have a list of required parameters needed to 
properly configure the Recipe.
The Recipe announces the required parameters with the following syntax 
(the syntax is subject to changes).

.. code-block:: python

    class SomeRecipe(RecipeBase):
        __requires__ = [
            Parameter('master_dark', MasterDark, 'Master dark image'),
            Parameter('some_numeric_value', 0.45, 'Some numeric value'),
        ]
        ...

When the reduction is run from the command line using Numina CLI, the program checks 
that the required values are provided or have default vales. When run the reduction is 
run automatically using Pontifex, the program searches the operation database searching 
for the most appropriated data products (in this case, a MasterDark frame).

When the Recipe is properly configured, it is executed with a observing block data 
structure as input. When run using Numina CLI, the data structure is created from a 
text file. When run with Pontifex, the data structure is created from the contents of
the database.
