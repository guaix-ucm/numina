************************
Numina Pipeline Concepts
************************

Instrument
##########


Observing Modes
###############
Each Instrument has a list of predefined types of observations that can be
carried out with it. Each Observing Mode is defined by:

  * The configuration of the Telescope
  * The configuration of the Instrument
  * The type of processing required by the images obtained durinf the observation

Some of the observing modes of a Instrument are **Scientific**, that is, modes devoted
to obtain data to perform scientific analysis. Other modes are devoted to **Calibration**;
these modes produce data required to correct the scientific images from the effects
of the Instrument, the Telescope and the atmosphere.

Recipes
#######
A recipe is a method to process the images obtained in a particular observing
mode. Recipes in general require (as inputs) the list of raw images obtained
during the observation. Recipes can require other inputs (calibrations), and
those inputs can be the outputs of other recipes.

Images obtained in a particular mode are processed by one recipe.

.. graphviz::

    digraph G {
        rankdir=LR;
        subgraph cluster_0 {
            style=filled;
            color=lightgrey;
            node [style=filled,color=white];
            edge[style=invis]
            a0 -> a1 -> a2 -> a3;
            #label = "Observing\nModes";
        }

        subgraph cluster_1 {
            node [style=filled];
            edge[style=invis]
            b0 -> b1 -> b2 -> b3;
            #label = "Recipes";
        }

        a0 -> b0 [rank=same];
        a1 -> b1 [rank=same];
        a2 -> b2 [rank=same];
        a3 -> b3 [rank=same];

        a0 [label="Mode 1"];
        a1 [label="Mode 2"];
        a2 [label="..."];
        a3 [label="Mode N"];
        b0 [label="Recipe 1"];
        b1 [label="Recipe 2"];
        b2 [label="..."];
        b3 [label="Recipe N"];

    }


Pipelines
#########
A pipeline represents a particular mapping between the observing modes and the
reduction algorithms that process each mode. Each instrument has at least one
pipeline called *default*. It may have other pipelines for specific purposes.


.. graphviz::

    digraph G {
        subgraph cluster_0 {
            style=filled;
            color=lightgrey;
            node [style=filled,color=white];
            edge[style=invis]
            a0 -> a1 -> a2 -> a3;
            label = "Observing\nModes";
        }

        subgraph cluster_1 {
            node [style=filled];
            edge[style=invis]
            b0 -> b1 -> b2 -> b3;
            label = "pipeline: \"default\"";
            color=blue
        }

        subgraph cluster_2 {
            node [style=filled];
            edge[style=invis]
            b11 -> b12 -> b13 -> b14;
            label = "pipeline: \"test\"";
            color=blue
        }

        a0 -> b0;
        a1 -> b1;
        a2 -> b2;
        a3 -> b3;
        a0 -> b11;
        a1 -> b12;
        a2 -> b13;
        a3 -> b14;

        a0 [label="Mode 1"];
        a1 [label="Mode 2"];
        a2 [label="..."];
        a3 [label="Mode N"];
        b0 [label="Recipe 1"];
        b1 [label="Recipe 2"];
        b2 [label="..."];
        b3 [label="Recipe N"];
        b11 [label="Recipe 11"];
        b12 [label="Recipe 12"];
        b13 [label="..."];
        b14 [label="Recipe M"];
    }


Products, Requirements and Data Types
#####################################
A recipe announces its required inputs as :class:`~numina.core.requirements.Requirement` and its outputs as
:class:`~numina.core.dataholders.Result`.

Both Results and Requirements have a name and a type. Types can be plain
Python types or defined by the developer.

Format of the input files
#########################

The default format of the input and output files is YAML_, a data
serialization language.

Format of the Observation Result file
*************************************
This file contains the result of an observation. It represents an
:class:`~numina.core.oresult.ObservationResult` object.

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

frames: required, list of file names
    List of raw images

.. code-block:: yaml

   id: 21
   instrument: EMIR
   mode: nb_image
   children: []
   frames:
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
******************************************
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
******************************
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

.. _YAML: http://www.yaml.org
