

Numina Pipeline Concepts
========================

Instrument
----------


Observing Modes
---------------
Each instrument has a list of predefined types of observations that can be
carried out with it. Some types of observations provide calibrations needed by
other types of observations.

Pipelines
---------
A pipeline represents a particular mapping between the observing modes and the
reduction algorithms that process each mode. Each instrument has at least one
pipeline.

Recipes
-------
A recipe is a method to process the images obtained in a particular observing
mode. Recipes in general require (as inputs) the list of raw images obtained 
during the observation. Recipes can require other inputs (calibrations), and 
those inputs can be the outputs of other recipes.

Products, Requirements and Data Types
-------------------------------------
A recipe announces its required inputs as Requirements and its outputs as
Products.

Both products and requirements have a name and a type. Types can be plain
Python types or defined by the developer.

Numina Pipeline Example
=======================

This is a Numina Pipeline for an instrument called CLODIA.

Observing Modes
---------------

CLODIA is a simple CCD camera, so it has only three modes called BIAS, FLAT
and SCIENCE. BIAS is used to obtain pedestal images, FLAT is used to obtain
flat-field images and SCIENCE is used to obtain images of scientific targets.

Pipelines
---------
In this simple example, there is only one pipeline, called 'default'. It is a
mapping between BIAS and its recipe (BiasRecipe), FLAT and FlatRecipe and
SCIENCE and ScienceRecipe.

Recipes
-------
We have three recipes: BiasRecipe, FlatRecipe, ScienceRecipe. Next we define
theirs inputs and outputs.

BiasRecipe
..........

This recipe requires the images obtained in BIAS mode. In a very simple
instrument without overscan, the recipe will just combine the raw images to
create a master bias image, so no other inputs are required.

This recipe produces an image. There is a predefined data type for images, but
we can add more information to the recipe by defining a type MasterBias.

FlatRecipe
..........
This recipe requires the images obtained in FLAT mode. It also requires an
image produced by BiasRecipe (to correct the pedestal in FLAT images), so we
add a requirement in MasterBias.

This recipes produces an image, but in this case is a MasterFlat. We define
again a new type called MasterFlat.


ScienceRecipe
.............
This recipe requires the images obtained in SCIENCE mode and two calibrations:
a MasterBias and a MasterFlat images.

This recipe produces an image, we can define a type for it, such as
ScienceImage.
