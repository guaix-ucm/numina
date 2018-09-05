
.. _cli:

======================
Command Line Interface
======================

The :program:`numina` script is the interface with the pipelines
It is called like this::

     $ numina [global-options] comands [comand-options]

The :program:`numina` script has several options:

.. program:: numina

.. option:: -d, --debug

   Debug enabled, increases verbosity.

.. option:: -l filename

   A file con configuration options for logging.

Options for run
===============
The run subcommand processes the observing result with the
appropriated reduction recipe.

It is called like this::

     $ numina [global-options] run [comand-options] observation-result.yaml

.. program:: numina run

.. option:: --instrument 'name'

   Name of one of the predefined instrument configurations.

.. option:: --pipeline 'name'

   Name of one of the predefined pipelines.
   
.. option:: --requirements filename

   File with the description of the parameters of the recipe.
      
.. option:: --basedir path

   File path used to resolve relative paths in the following options.
   
.. option:: --datadir path

   File path to the folder containing the pristine data to be processed.
   
.. option:: --resultsdir path

   File path to the directory where results are stored.

.. option:: --workdir path

   File path to the a directory where the recipe can write. Files in datadir
   are copied here.
   
.. option:: --cleanup

   Remove intermediate and temporal files created by the recipe.
   
.. option:: observing_result filename

   Filename contaning the description of the observation result.

Options for show-instruments
============================
The show-instruments subcommand outputs information about the instruments
with available pipelines.

It is called like this::

     $ numina [global-options] show-instruments [options] 

.. program:: numina show-instruments

.. option:: -o, --observing-modes

   Show names and keys of Observing Modes in addition of instrument
   information.

.. option:: name

   Name of the instruments to show. If empty show all instruments.
   
Options for show-modes
======================
The show-modes subcommand outputs information about the observing
modes of the available instruments.

It is called like this::

     $ numina [global-options] show-modes [options] 

.. program:: numina show-modes

.. option:: -i, --instrument name

   Filter modes by instrument name.

.. option:: name

   Name of the observing mode to show. If empty show all observing modes.
   
Options for show-recipes
========================
The show-recipes subcommand outputs information about the recipes
of the available instruments.

It is called like this::

     $ numina [global-options] show-recipes [options] 

.. program:: numina show-recipes

.. option:: -i, --instrument name

   Filter recipes by instrument name.

.. option:: -t, --template

   Generate a template file to be used a requirement file 
   by :program:`numina run`. 

.. option:: -m, --mode

   Show the recipe that matches the name of the mode

.. option:: name

   Name of the recipe to show. If empty show all recipes.
   
