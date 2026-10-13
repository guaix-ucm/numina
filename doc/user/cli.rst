
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

   A file with configuration options for logging.

Options for run
===============
The run subcommand processes the observing result with the
appropriated reduction recipe.

It is called like this::

     $ numina [global-options] run [comand-options] observation-result.yaml

.. program:: numina run

.. option:: --insconf, --profile UUID

   UUID of one of the instrument configurations. It overrides the
   configuration selected from the images (keyword INSCONF).

.. option:: --profile-path path

   Directory with additional instrument configurations.

.. option:: -p, --pipeline 'name'

   Name of one of the pipelines of the DRP. It overrides the pipeline
   of the observation result, the default is 'default'.

.. option::  -r, --requirements filename

   File with the description of the parameters of the recipe
   (control file, in format 1). It is validated with the schema
   ``control-schema.json`` of :mod:`numina.schemas`, the errors
   show where the file is not valid.

.. option:: --basedir path

   File path used to resolve relative paths in the following options.

.. option:: --datadir path

   File path to the folder containing the pristine data to be processed.

.. option:: --db FILE

   Registry of the reductions, relative to basedir. It is created if it
   does not exist. The tasks, results and products of the reductions are
   recorded in it, and later reductions use them: a product is the most
   recent one in the registry with quality control different from BAD,
   for the same instrument, profile and tags. The values given with
   ``--parameter-NAME=VALUE`` and in the requirements of the observation
   result have priority over the registry, and the products of the control
   file and of calibsdir are used if there is none in the registry.
   It overrides the value of ``file`` in the section
   ``[tool.db]`` of the configuration. With the registry, the templates of
   the directories are read from ``[tool.db]``, and they include the id of
   the task.

.. option:: --copy-files

   Copy the files of the observation result and the requirements
   to the work directory.

.. option:: --link-files, --not-copy-files

   Link the files of the observation result and the requirements
   in the work directory. Without --copy-files or --link-files, the
   value of copy_files in the configuration is used, by default
   the files are linked.

.. option:: -e, --enable BLOCKID

   enable a block listed in the observation result

.. option:: --request FILE

   File with reduction requests: the requirements, pipeline, profile or
   mode used to reduce each observing block (see
   :ref:`reduction-requests`).

.. option:: --validate

   Validate the inputs and the results of the recipe. Without this
   option, the value of validate in ``[tool.run]`` of the configuration
   is used, by default the inputs and results are not validated.

   Each input is validated with its type. For the observation result,
   each raw image is checked with the checker of the DRP of the
   instrument, through :data:`numina.core.config.check` (the same
   used by ``numina verify``), as the raw image type of the observing
   mode (``rawimage`` in ``drp.yaml``), and then the observation result
   is checked with the validator of the mode (``validator`` in
   ``drp.yaml``). If the DRP has no checker, the raw images are not
   checked.

   If an input is not valid, the reduction stops before running the
   recipe, with an error that lists the invalid inputs. The results
   are validated after the recipe runs.

.. option:: observing_result filename

   Filename containing the description of the observation result.

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

.. option:: -m, --mode

   Filter recipes by observing mode.

.. option:: name

   Name of the recipe to show. If empty show all recipes.

Options for verify
==================
The verify subcommand checks files with the checker of the DRP of their
instrument, through :data:`numina.core.config.check`, the same used by
``numina run --validate``. It is useful to discard raw images with
incomplete headers before reducing them.

It is called like this::

     $ numina [global-options] verify [options] files

Each file is reported in one line, as ``OK``, ``INVALID`` (with the reason)
or ``NOT CHECKED`` (the instrument has no checks, the image has no
``INSTRUME``, or the file is not a FITS or JSON file), followed by a summary::

    $ numina verify r0001.fits r0002.fits
    OK r0001.fits
    INVALID r0002.fits: HDU 0: 'VPH' is a required property
    1 valid, 1 invalid, 0 not checked

The exit status is 1 if some file is invalid, and 0 otherwise.

.. program:: numina verify

.. option:: --mode MODE

   Check the images as the raw images of the observing mode MODE (or
   INSTRUMENT.MODE, if several DRPs have a mode with that name), instead
   of the type deduced from their headers.

.. option:: --ob

   The files are observation results. Each raw image is checked as the raw
   image of the observing mode, and then the observation result with the
   validator of the mode, as with ``numina run --validate``. Not allowed
   with :option:`--mode`.

.. option:: --datadir path

   Directory of the raw images of the observation results. By default,
   the value of datadir in ``[tool.run]`` of the configuration.

.. option:: files

   The files to check: FITS images and JSON files or, with :option:`--ob`,
   observation results.

Options for db
==============
The db subcommand works with the registry of reductions, the file given
with ``--db`` or with ``file`` in the section ``[tool.db]`` of the
configuration. It has these subcommands.

``numina db ingest`` records the observing blocks of the observation files
in the registry, without reducing them::

     $ numina db ingest --db numina-db.json 0_bias.yaml 1_tracemap.yaml
     new oblock_id=0_bias
     new oblock_id=1_HR-R

Each block is reported as ``new``, ``updated`` (its definition has
changed) or ``unchanged``.

``numina db run`` reduces observing blocks of the registry by their id,
in the order given::

     $ numina db run --db numina-db.json -r control.yaml 0_bias 1_HR-R

It accepts the options of ``numina run`` that describe how to reduce,
like ``-r``, ``--profile``, ``--pipeline``, ``--validate`` or
``--parameter-NAME=VALUE``. The exit status is 2 if a block is not in the
registry. The children of a block are read from the registry too.

With ``--request FILE``, the ids are optional: the blocks reduced are those
of the requests with id, in the order of the file.

``numina db list`` prints a table with the observing blocks, tasks,
results or products of the registry, one line for each document. In the
table, the uuids are abbreviated to their first 8 characters, as accepted
by the references to the registry, and the times are shown to the second::

     $ numina db list --db numina-db.json products --tags
     id  uuid      type               oblock_id  instrument  qc    priority  time_obs             content
     2   a5fb2b92  TraceMap           1_HR-R     MEGARA      GOOD  0         2017-08-31T18:09:14  obsid1_HR-R_2_results/master_traces.json
         └─ insmode=LCB temp=15.91 vph=HR-R
     7   f691f538  MasterSensitivity  7_HR-R     MEGARA      GOOD  0         2017-07-25T03:50:28  obsid7_HR-R_8_results/master_sensitivity.fits
         └─ (no tags)

.. program:: numina db list

.. option:: oblocks, tasks, results, products

   The collection to show.

.. option:: --oblock-id ID, --instrument NAME, --mode MODE

   Show only the documents with this value. A collection without the field
   is an error.

.. option:: --tag KEY=VALUE

   Show only the products with this tag. It can be repeated, to select
   the products with all the tags.

.. option:: --tags

   Show the tags of each product, in a line below it. Without this
   option, there is one line for each document, that can be processed
   with tools like grep.

.. option:: --json

   Print the complete documents of the registry, with all their fields
   (the tags, the uuids, the inputs of the tasks...), in JSON. It can be
   processed with tools like jq, for example::

       numina db list products --json | jq '.[] | select(.tags.vph == "HR-R") | .content'

``numina db priority`` sets the priority of a product, by its id or its
uuid (that can be abbreviated)::

     $ numina db priority --db numina-db.json f5dea9e0 1
     product 1 has priority 1

The automatic selection of products in the registry uses, among the
products with quality control different from ``BAD`` that are valid for
the observing block, the product with the highest priority (0 by default,
it can be negative), then the one observed closest to the date of the
first image of the block, then the most recent. The priorities are shown
by ``numina db list products``.

The subcommands accept ``--db FILE``, ``--basedir path`` (the registry and
its paths are relative to it) and ``--datadir path``.

.. _reduction-requests:

Reduction requests
==================
An observing block describes the images to reduce. How to reduce them can
be given apart, in a file of reduction requests, with ``--request FILE``
in ``numina run`` and ``numina db run``, or with the argument ``request``
of :meth:`numina.user.session.Session.run`. The same block can be reduced
with different requests, for example with two different calibrations.

Each document of the file is a request, with these keys, all optional:

``id``
   The id of the observing block. A request without id applies to the
   blocks without their own request. There can be only one.

``requirements``
   Values of the requirements of the recipe, by name, as in the
   ``requirements`` of an observing block. The paths of the files are
   relative to the directory of the data (``--datadir``).

``pipeline``, ``profile``, ``mode``
   The pipeline, the UUID of the instrument configuration and the
   observing mode used to reduce the block, instead of those of the
   block.

For example::

    id: 3_HR-R
    requirements:
      master_bias: /data/calibrations/master_bias_2017.fits
    ---
    requirements:
      reference_column: 2100

The files are validated with ``request-schema.json``
(:mod:`numina.schemas`).

The value of a requirement is searched, in this order, in:

1. the command line (``--parameter-NAME=VALUE``);
2. the reduction request;
3. the observing block;
4. the results of other blocks, for the requirements that use them;
5. for products, the registry of reductions, the control file and the
   directory of calibrations; for parameters, the control file;
6. the default value of the requirement.

The ``--pipeline`` and ``--profile`` options of the command line override
those of the requests.

Each task records where the value of each requirement was found, and the
value (a file name, for example), in ``inputs`` of its ``request_runinfo``,
in the file ``task.json`` and in the registry of reductions. For the
products and results of the registry, their id is recorded too, and for
the references, the reference and where it was given (the request or the
observing block).

References to the registry
--------------------------
With a registry of reductions, the value of a requirement, in a request or
in an observing block, can be a reference to a product or a result of the
registry, instead of a file name::

    requirements:
      master_bias: {product: f5dea9e0}
      master_wlcalib: {result_of: 12_arc}
      master_traces: {result: 9b2c41d7, field: master_traces}

``product``
   A product, by its id (an integer) or its uuid (a string), as shown by
   ``numina db list products``.

``result``
   A result, by its id or its uuid, as shown by ``numina db list results``.

``result_of``
   The most recent result of an observing block, with quality control
   different from ``BAD``. It changes if the block is reduced again.

``task``
   The result of a task, by its id, as shown by ``numina db list tasks``.
   It is an error if the task failed or has not finished.

The uuids can be abbreviated, with at least their first 8 characters
(``{product: f5dea9e0}``); it is an error if the abbreviation is
ambiguous. The ids of products, results and tasks are local to the
registry: the same product has other id in other registry, or if the
registry is created again. The uuids are part of the products and results,
in their headers and result files. Use the uuids in the files of requests
that are kept, and the ids when reducing interactively.

For results, ``field`` is the field of the result, by default the name of
the requirement. The value is a file of the result (a product, for
example), or the value stored in the result file (a number, for example).

The references choose a calibration explicitly, instead of the automatic
selection of the registry (see ``numina db priority``). A product or result
chosen explicitly is used even if its quality control is ``BAD``, with a
warning. It is an error if what they refer to is not in the registry, if
the product is not of the type of the requirement, if its tags do not match
those of the observing block (as in the automatic selection, for example a
trace map of other VPH), or if there is no registry. These errors stop the
reduction before running the recipe.

Options for inittree
====================
The inittree subcommand installs a dataset of calibrations of an
instrument (master bias, trace maps, lines catalogues...), declared by its
DRP, in the directory of calibrations used by ``numina run``::

     $ numina inittree
     MEGARA  megaradrp-calibrations-2026.2  Calibrations of MEGARA, release 2026.2
     $ numina inittree MEGARA
     dataset megaradrp-calibrations-2026.2 of MEGARA, from https://zenodo.org/records/...
     58 new files, 0 unchanged, 0 with other contents, in calibrations/MEGARA
     installed 58 files in calibrations/MEGARA
     calibsdir set to calibrations in /home/user/work/.numina.cfg

The dataset, a ZIP file, is downloaded to the cache of numina (see
:mod:`numina.util.download`), and its hash is checked. If all the files of
the ZIP file are under one directory, that directory is removed from their
paths, and the hidden files are not installed. Before installing anything,
the files are compared with those that already exist: the files with the
same contents are left as they are, and if some file has other contents,
nothing is installed, unless ``--overwrite`` is given.

``calibsdir`` is set in the section ``[tool.run]`` of the file
``.numina.cfg`` of the current directory, and the section
``[tool.inittree.INSTRUMENT]`` records the dataset installed.

.. program:: numina inittree

.. option:: instrument

   The instrument. Without it, the datasets declared by the DRPs are
   listed.

.. option:: --dataset NAME

   The dataset to install, by default the last one declared by the DRP.

.. option:: --base-path path

   The directory of calibrations, ``calibrations`` by default. The files
   are installed under ``path/INSTRUMENT``.

.. option:: --overwrite

   Replace the files that exist with other contents.

.. option:: --dry-run

   Show what would be done, without downloading to the directory of
   calibrations or changing any file (the dataset is downloaded to the
   cache).

.. option:: --no-config

   Do not change ``.numina.cfg``.

A DRP declares its datasets in the key ``calibrations`` of its
``drp.yaml``, with the URL of the ZIP file and its hash (``md5``,
``sha1`` or ``sha256``, as ``algorithm:value``)::

    calibrations:
    - name: megaradrp-calibrations-2026.2
      description: Calibrations of MEGARA, release 2026.2
      url: https://zenodo.org/records/18623771/files/guaix-ucm/megaradrp-calibrations-2026.2.zip
      known_hash: md5:3d98803089d4822ee00ed03a4dbba229

The key ``directory`` changes the directory of the dataset under the
directory of calibrations, the name of the instrument by default.
