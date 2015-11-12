
Numina Pipeline Environment
###########################

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

Users of the DRP will use the simple Numina CLI.
Users of the DFP shall interact with the software
through the GTC Inspector.

Recipe Requirements and Products
--------------------------------
Recipes based on Numina have a list of requirements needed to 
properly configure the Recipe.
Recipes also provide a list of products created by the recipe.

The Recipe announces its requirements and producst with the following syntax.

.. code-block:: python

    class SomeRecipe(RecipeBase):        

        master_dark = Requirement(MasterDark, 'Master dark image')
        some_numeric_value = Parameter(0.45, 'Some numeric value'),

        master_flat = Product(MasterDark) 

When the reduction is run from the command line using Numina CLI, the program 
checks that the required values are provided or have default values. 

When the Recipe is properly configured, it is executed with a observing block 
data structure as input. When run using Numina CLI, the data structure is 
created from a text file.

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

images: required, list of file names
    List of raw images

.. code-block:: yaml

   id: 21
   instrument: EMIR
   mode: nb_image
   children: []
   images:
   - r0121.fits
   - r0122.fits
   - r0123.fits
   - r0124.fits
   - r0125.fits
   - r0126.fits
   - r0127.fits
   - r0128.fits
   - r0129.fits
   - r0130.fits
   - r0131.fits
   - r0132.fits

Format of the requirement file (version 1)
''''''''''''''''''''''''''''''''''''''''''
.. code-block:: yaml

    version: 1
    products:
      EMIR:
       - {id: 1, content: 'file1.fits', type: 'MasterFlat', tags: {'filter': 'J'}, ob: 200} 
       - {id: 4, content: 'file4.fits', type: 'MasterBias', tags: {'readmode': 'cds'}, ob: 400}
      MEGARA:
       - {id: 1, content: 'file1.fits', type: 'MasterFlat', tags: {'vph': 'LR1'}, ob: 1200}
       - {id: 2, content: 'file2.yml', type: 'TraceMap', tags: {'vph': 'LR2', 'readmode': 'fast'}, ob: 1203}
    requirements:
      EMIR:
        default:
           TEST6: 
              pinhole_nominal_positions: [ [0, 1], [0 , 1]]
              box_half_size: 5
           TEST9:
              median_filter_size: 5
    MEGARA:
        default: 
           mos_image: {}


Format of the requirement file 
''''''''''''''''''''''''''''''
.. warning::
   This section documents a deprecated format

.. deprecated:: 0.14.0

This file contains configuration parameters for the recipes that
are not related to the particular instrument used.

The contents of the file are serialized as a dictionary with the
following keys:

requirements: required, dictionary
    A dictionary of parameter names and values.

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
