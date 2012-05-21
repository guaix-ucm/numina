
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

     $ numina [global-options] run [comand-options] task

.. program:: numina run

.. option:: --instrument filename

   File with the instrument description
   
.. option:: --obsblock filename

   File with the observing block description
      
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
   
.. option:: task filename

   Filename of a file contaning the parameters of the reduction    

Options for list
================
The list subcommand lists all the recipes available in the system.

It is called like this::

     $ numina [global-options] list

Options for list_instrument
===========================
The list_instrument subcommand lists all the instrumetns available in the system.

It is called like this::

     $ numina [global-options] list_instrument
