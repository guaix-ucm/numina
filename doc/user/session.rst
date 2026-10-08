
.. _session:

=====================================
Reductions from scripts and notebooks
=====================================

The class :class:`numina.user.session.Session` reduces observing blocks
from Python, as :program:`numina run` does from the command line::

    from numina.user.session import Session

    session = Session(basedir=".", control="control.yaml", db="numina-db.json")
    session.add_observations("0_bias.yaml", "1_tracemap.yaml")

    task = session.run("0_bias")
    master_bias = task.result.master_bias.open()

    task = session.run("1_HR-R")

The arguments of :class:`~numina.user.session.Session` are:

``basedir``
   Base directory, where the work and result directories are created.
   The other paths are relative to it. The current directory by default.

``datadir``
   Directory with the raw data, ``data`` by default.

``control``
   Control file (in format 1), with the parameters and the precomputed
   products of the reduction. Optional.

``db``
   File of the registry of reductions. Optional.

``config``
   Configuration of numina. By default, the same configuration used by
   :program:`numina run`: the defaults, ``~/.config/numina/numina.cfg``
   and ``.numina.cfg`` in the current directory.

:meth:`~numina.user.session.Session.add_observations` adds observing
blocks, from YAML files (with one or several observing blocks) or as
dictionaries. :meth:`~numina.user.session.Session.run` reduces one of them
and returns the task, with the result of the recipe in ``task.result``.

Registry of reductions
======================

With ``db``, the tasks, results and products of the reductions are
recorded in a file, created if it does not exist. The following reductions,
in the same session or in later ones, use the recorded products: for
example, the master bias reduced in a session is used by the reductions of
a later session, without listing it in the control file.

A product is the most recent one in the registry with quality control
different from ``BAD``, for the same instrument, instrument profile and
tags. The values in the requirements of the observing block and those given
with ``--parameter-NAME=VALUE`` have priority over the registry, and the
products in the control file and in ``calibsdir`` are used if there is none
in the registry.

The registry can be queried::

    session.products(type="MasterBias")
    session.tasks(oblock_id="0_bias")
    session.results(oblock_id="1_HR-R")

With a registry, each task has its own work and result directories,
``obsid<id_of_obs>_<id_of_task>_work`` and
``obsid<id_of_obs>_<id_of_task>_results`` by default (templates in the
section ``[tool.db]`` of the configuration).

The registry is used by :program:`numina run` with ``--db FILE``.

The observing blocks are not restored from the registry: they must be added
again in each session with :meth:`~numina.user.session.Session.add_observations`.
