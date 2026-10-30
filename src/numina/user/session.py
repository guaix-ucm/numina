#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Reduction sessions, to use numina from scripts and notebooks

A :class:`Session` reduces observing blocks as ``numina run`` does::

    from numina.user.session import Session

    session = Session(basedir=".", control="control.yaml", db="numina-db.json")
    session.add_observations("0_bias.yaml", "1_tracemap.yaml")
    task = session.run("0_bias")
    task.result.master_bias

With a registry of reductions (``db``), the products and results of
previous reductions, in this or in previous sessions, are used by the
following ones.
"""

import os

from numina.schemas import validate as validate_schema

from .baserun import run_reduce
from .cli import load_config
from .helpers import ReductionRequests, create_datamanager, load_observations


class Session:
    """A session of reductions

    Parameters
    ----------
    basedir : str or path-like
        Base directory, where the work and result directories are created.
        The other paths are relative to it.
    datadir : str or path-like
        Directory with the raw data
    control : str or path-like, optional
        Control file, in format 1
    db : str or path-like, optional
        File of the registry of reductions, created if it does not exist.
        Without a registry, the reductions are not recorded.
    config : configparser.ConfigParser, optional
        Configuration of numina. By default, the same configuration
        used by ``numina run`` (see :func:`numina.user.cli.load_config`)
    profile_path : str or path-like, optional
        Directory with additional instrument configurations
    """

    def __init__(self, basedir=".", datadir="data", control=None, db=None, config=None, profile_path=None):
        self.basedir = os.path.abspath(basedir)
        self.config = load_config() if config is None else config
        section = self.config["tool.run"]
        section["basedir"] = self.basedir
        section["datadir"] = str(datadir)
        if db is not None:
            if not self.config.has_section("tool.db"):
                self.config.add_section("tool.db")
            self.config["tool.db"]["file"] = str(db)
        reqfile = None if control is None else self._path(control)
        #: The DataManager used by the reductions
        self.datamanager = create_datamanager(self.config, reqfile, profile_path_extra=profile_path)

    def _path(self, name):
        """Return the path of ``name`` relative to the base directory."""
        return os.path.join(self.basedir, name)

    @property
    def registry(self):
        """The registry of reductions (numina.dal.registry.Registry), or None"""
        return self.datamanager.registry

    def add_observations(self, *observations):
        """Add observing blocks

        Each argument is a YAML file (relative to basedir) with one or
        several observing blocks, or an observing block as a dictionary.
        """
        loaded = []
        for obs in observations:
            if isinstance(obs, dict):
                loaded.append(obs)
            else:
                _, loaded_obs = load_observations([self._path(obs)])
                loaded.extend(loaded_obs)
        self.datamanager.backend.add_obs(loaded)

    def run(self, obsid, request=None, **kwargs):
        """Reduce an observing block, return the task

        The keyword arguments are those of :func:`numina.user.baserun.run_reduce`
        (as_mode, pipeline, profile, requirements, copy_files, validate_inputs,
        validate_results, strict_inputs). The defaults of copy_files, and of
        validate_inputs and validate_results, are the values of copy_files and
        validate in ``[tool.run]`` of the configuration, as in ``numina run``.

        `request` is a reduction request, a YAML file (relative to basedir)
        with requests, or a request as a dictionary (requirements, pipeline,
        profile, mode). The keyword arguments override its values.
        """
        if request is not None:
            if isinstance(request, dict):
                validate_schema(request, "request", source="request")
                requests = ReductionRequests(default=request)
            else:
                requests = ReductionRequests.load(self._path(request))
            for key, value in requests.arguments(obsid).items():
                kwargs.setdefault(key, value)
        section = self.config["tool.run"]
        kwargs.setdefault("copy_files", section.getboolean("copy_files"))
        validate = section.getboolean("validate")
        kwargs.setdefault("validate_inputs", validate)
        kwargs.setdefault("validate_results", validate)
        return run_reduce(self.datamanager, obsid, **kwargs)

    def set_priority(self, product, priority):
        """Set the priority of a product of the registry, by id (integer) or uuid

        The automatic selection of products uses the product with the
        highest priority (0 by default), and then the one observed closest
        to the observing block. Returns the id of the product.
        """
        return self._require_registry().set_priority(product, priority)

    def _require_registry(self):
        """Return the registry of reductions.

        Raises
        ------
        ValueError
            If the session has no registry.
        """
        if self.registry is None:
            raise ValueError("the session has no registry of reductions, use Session(db=...)")
        return self.registry

    def tasks(self, **equal):
        """Tasks in the registry, filtered by equality of fields"""
        return self._require_registry().tasks(**equal)

    def results(self, **equal):
        """Results in the registry, filtered by equality of fields"""
        return self._require_registry().results(**equal)

    def products(self, **equal):
        """Products in the registry, filtered by equality of fields"""
        return self._require_registry().products(**equal)
