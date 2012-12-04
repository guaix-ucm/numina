
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
**Data Factory Pipeline**, (:term:`DFP`).

Other group of recipes are devoted to scientific observing modes: imaging, 
spectroscopy and auxiliary calibrations. These Recipes constitute the
**Data Reduction Pipeline**, (:term:`DRP`). The software is meant to be standalone,
users shall download the software and run it in their own computers, with
reduction parameters and calibrations provided by the instrument team.

Users of the DRP may use the simple Numina CLI or the higher level,
database-driven Pontifex. Users of the DFP shall interact with the software
through the GTC Inspector.

Recipe Requirements
-------------------
Recipes based on Numina have a list of requirements needed to 
properly configure the Recipe.
The Recipe announces its requirements with the following syntax 
(the syntax is subject to changes).

.. code-block:: python

    class SomeRecipeRequirements(RecipeRequirements):
        master_dark = DataProductRequirement(MasterDark, 'Master dark image') 
        some_numeric_value = Parameter(0.45, 'Some numeric value'),

    @define_requirements(SomeRecipeRequirements)
    class SomeRecipe(RecipeBase):        
        ...

When the reduction is run from the command line using Numina CLI, the program 
checks that the required values are provided or have default vales. 
When the reduction is run automatically using Pontifex, the program searches 
the operation database for the most appropriated data products 
(in this case, a MasterDark frame).

When the Recipe is properly configured, it is executed with a observing block 
data structure as input. When run using Numina CLI, the data structure is 
created from a text file. When run with Pontifex, the data structure is 
created from the contents of the database.

Recipe Products
--------------- 
Recipes based on Numina provide a list of products created by the recipe.
The Recipe announces the required parameters with the following syntax 
(the syntax is subject to changes).

.. code-block:: python

    class SomeRecipeRequirements(RecipeRequirements):
        master_dark = DataProductRequirement(MasterDark, 'Master dark image') 
        some_numeric_value = Parameter(0.45, 'Some numeric value'),

    class SomeRecipeResult(RecipeResult):
        master_flat = Product(MasterDark) 
        
    @define_requirements(SomeRecipeRequirements)
    @define_result(SomeRecipeResult)
    class SomeRecipe(RecipeBase):        
        ...


Format of the input files
-------------------------

The default format of the input and output files is YAML_, a data 
serialization language. 

Format of the Observation Result file
'''''''''''''''''''''''''''''''''''''
This file contains the result of a observation. It represents a 
:class:`numina.recipes.oblock.ObservationResult` object.

The contents of the object are serialized as a dictionary with the
following keys:

id: not required, integer, defaults to 1
    Unique identifier of the observing block

instrument: required, string
    Name of the instrument, as it appears in the instrument file
    (see below)

mode: required, string
    Name of the observing mode

children: not required, list of integers, defaults to empty list
    Identifications of nested observing blocks

frames: required, list of frame-info
    List of frames

Additionally, the frame-info is defined as follows:

frame-info: list of 2 strings
    A list of two strings, first is the FITS file of the frame, 
    second is the type of frame

.. code-block:: yaml

   id: 21
   instrument: EMIR
   mode: nb_image
   children: []
   frames:
   - [r0121.fits, TARGET]
   - [r0122.fits, TARGET]
   - [r0123.fits, TARGET]
   - [r0124.fits, SKY]
   - [r0125.fits, SKY]
   - [r0126.fits, SKY]
   - [r0127.fits, TARGET]
   - [r0128.fits, TARGET]
   - [r0129.fits, TARGET]
   - [r0130.fits, SKY]
   - [r0131.fits, SKY]
   - [r0132.fits, SKY]

Format of the instrument file
'''''''''''''''''''''''''''''
.. warning::
   The instrument file is not needed in numina 0.9 and later. 

.. deprecated:: 0.9.0

This file contains configuration parameters for the recipes that
are related to the instrument. This information is not likely
to change in a short time basis. 

The contents of the file are serialized as a dictionary with the
following keys:

name: required, string
    Name of the instrument

pipeline: required, string
    Name of the pipeline that will process the data taken with the 
    instrument

keywords: optional, dictionary, defaults to {}
    A dictionary of keys and FITS keywords


The file may contain additional keys.

.. code-block:: yaml

    name: EMIR
    pipeline: emir
    keywords: {airmass: AIRMASS, exposure: EXPTIME, imagetype: IMGTYP, juliandate: MJD-OBS}
    detector:
      shape: [2048, 2048]

Format of the requirement file
''''''''''''''''''''''''''''''
This file contains configuration parameters for the recipes that
are not related to the particular instrument used.

The contents of the file are serialized as a dictionary with the
following keys:

requirements: required, dictionary
    A dictionary of parameter names and values.

parameters: deprecated, dictionary
    A dictionary of parameter names and values.

    Used only if requirements is not present.

products: optional, dictionary
    A dictionary with names for the products

logger: optional, dictionary
    A dictionary used to configure the custom file logger

.. code-block:: yaml

   requirements:
     master_bias: master_bias-1.fits
     master_bpm: bpm.fits
     master_dark: master_dark-1.fits
     master_intensity_ff: master_flat.fits
     nonlinearity: [1.0, 0.0]
     subpixelization: 4
     window:
     - [800, 1500]
     - [800, 1500]
   products:
     flatframe: 'master_intensity_flat.fits'
   logger:
     logfile: processing.log
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     enabled: true

Generating template requirement files
-------------------------------------
Template requirement files can be generated by :program:`numina show-recipes`
The flag generates templates for the named recipe or for all the available
recipes if no name is passed. 

For example::

  $ numina show-recipes -t emir.recipes.DitheredImageRecipe
  # This is a numina 0.9.0 template file
  # for recipe 'emir.recipes.DitheredImageRecipe'
  #
  # The following requirements are optional:
  #  sources: None
  #  master_bias: master_bias.fits
  #  offsets: None
  # end of optional requirements
  requirements:
    check_photometry_actions: [warn, warn, default]
    check_photometry_levels: [0.5, 0.8]
    extinction: 0.0
    iterations: 4
    master_bpm: master_bpm.fits
    master_dark: master_dark.fits
    master_intensity_ff: master_intensity_ff.fits
    nonlinearity: [1.0, 0.0]
    sky_images: 5
    sky_images_sep_time: 10
  #products:
  # catalog: None
  # frame: frame.fits
  #logger:
  # logfile: processing.log
  # format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  # enabled: true
  ---

The # character is a comment, so every line starting with it can safely 
removed. The names of FITS files in requirements must be edited to point
to existing files.

.. _YAML: http://www.yaml.org
