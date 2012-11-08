
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

   Debug enabled

.. option:: -l filename

   Logging file


Options for run
===============
The run subcommand processes the observing result with the
appropriated reduction recipe.

It is called like this::

     $ numina [global-options] run [comand-options] observation-result

.. program:: numina run

.. option:: --instrument filename

   File with the instrument description
   
.. option:: --parameters filename

   File with the description of the parameters of the recipe
      
.. option:: --basedir path

   File path used to resolve relative paths in the following options
   
.. option:: --datadir path

   File path to the folder containing the data to be processed
   
.. option:: --resultsdir path

   File path to the directory where results are stored

.. option:: --workdir path

   File path to the a directory where the recipe can write files
   
.. option:: --cleanup

   Remove intermediate and temporal files created by the recipe
   
.. option:: observing_result filename

   Filename contaning the description of the observation result

Options for show
================
The show subcommand outputs information about the loaded pipelines

It is called like this::

     $ numina [global-options] show [show-options] 

.. program:: numina show

.. option:: -o, --observing-modes

   Show Observing Modes

.. option:: -r [id]

   Show Recipe whose identificator is id. If not listed, shows all
   recipes
   
.. option:: -i

   Show Instruments

