
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

Recipe Parameters
-----------------
Recipes based on Numina have a list of required parameters needed to 
properly configure the Recipe.
The Recipe announces the required parameters with the following syntax 
(the syntax is subject to changes).

.. code-block:: python

    class SomeRecipe(RecipeBase):
        __requires__ = [
            DataProductParameter('master_dark', MasterDark, 'Master dark image'),
            Parameter('some_numeric_value', 0.45, 'Some numeric value'),
        ]
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

Format of the parameter file
'''''''''''''''''''''''''''''
This file contains configuration parameters for the recipes that
are not related to the particular instrument used.

The contents of the file are serialized as a dictionary with the
following keys:

parameters: required, dictionary
    A dictionary of parameter names and values

.. code-block:: yaml

   parameters:
     master_bias: master_bias-1.fits
     master_bpm: bpm.fits
     master_dark: master_dark-1.fits
     master_intensity_ff: master_flat.fits
     nonlinearity: [1.0, 0.0]
     subpixelization: 4
     window:
     - [800, 1500]
     - [800, 1500]

Editing files
-------------

Altougth YAML files are plain text and can be easily red and edited by hand,
for mass edition and changing, we recommend using a YAML library.

For example, to create a instrument file using Python, we first create
the dictionary structure and finally we dump it with YAML::

  >>> import YAML
  >>> d = {}
  >>> d['name'] = 'EMIR'
  >>> d['pipeline'] = 'emir'
  >>> d['keywords'] = {}
  >>> d['keywords']['filter'] = 'FILTER'
  # Dumping to a file
  >>> with open('instrument.yaml', 'w') as fd:
  ...   yaml.dump(d, fd)


.. _yaml: http://www.yaml.org
