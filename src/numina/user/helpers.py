#
# Copyright 2008-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import datetime
import errno
import hashlib
import logging
import os
import shutil

import yaml

import numina.drps
from numina.dal.dictdal import HybridDAL
from numina.dal.registry import Registry
from numina.schemas import validate as validate_schema
from numina.dal.utils import PRODUCT_TEMPLATE_FIELDS, check_template, fill_template
from numina.util.jsonencoder import ExtEncoder
from numina.types.frame import DataFrameType
from numina.types.qc import QC
from numina.util.context import working_directory

_logger = logging.getLogger(__name__)

# Maximum difference, in seconds, between the modification times of a file
# and its copy to consider them equal, as the option --modify-window of rsync
MODIFY_WINDOW = 1.0


class DataManager:
    """Manager of the directories, files and records of the reductions.

    Parameters
    ----------
    basedir : str
        Base directory, where the work and result directories are created.
    datadir : str
        Directory of the raw data.
    backend : numina.dal.dictdal.HybridDAL
        DAL with the observing blocks, the recipes and their inputs.
    registry : numina.dal.registry.Registry, optional
        Registry of the reductions.
    """

    def __init__(self, basedir, datadir, backend, registry=None):
        self.basedir = basedir
        self.datadir = datadir
        self.backend = backend
        #: Registry of the reductions (numina.dal.registry.Registry), or None
        self.registry = registry

        # Templates of the names of directories and files, create_datamanager
        # replaces them with the values in the configuration
        self.workdir_tmpl = "obsid{obsid}_{taskid}_work"
        self.resultdir_tmpl = "obsid{obsid}_{taskid}_result"
        self.resultfile_tmpl = "result.json"
        self.taskfile_tmpl = "task.json"
        # the base of the names of the files of the products
        self.product_tmpl = "{name}"

    def set_templates(self, workdir_tmpl, resultdir_tmpl, resultfile_tmpl, taskfile_tmpl, product_tmpl="{name}"):
        """Set the templates of the names of directories and files

        Raises
        ------
        ValueError
            If a template uses fields not in numina.dal.utils.TEMPLATE_FIELDS
            (numina.dal.utils.PRODUCT_TEMPLATE_FIELDS for product_tmpl)
        """
        self.workdir_tmpl = check_template(workdir_tmpl)
        self.resultdir_tmpl = check_template(resultdir_tmpl)
        self.resultfile_tmpl = check_template(resultfile_tmpl)
        self.taskfile_tmpl = check_template(taskfile_tmpl)
        self.product_tmpl = check_template(product_tmpl, PRODUCT_TEMPLATE_FIELDS)

    def serializer(self, data, fd):
        """Write ``data`` in JSON to the file object ``fd``, with ``ExtEncoder``."""
        import json

        json.dump(data, fd, indent=2, cls=ExtEncoder)

    def store_result_to(self, result, obsid=None, taskid=None):
        """Save the products of the result, with the names of product_tmpl"""
        saveres = result.store_to(None, product_tmpl=self.product_tmpl, fields=dict(obsid=obsid, taskid=taskid))
        return saveres

    def restore_oblock(self, oblock_id):
        """Load the observing block `oblock_id` from the registry, if needed

        The observing blocks loaded from files have precedence: the block
        is read from the registry only if it is not loaded. Its children
        are restored too.

        Raises
        ------
        ValueError
            If the block is not loaded and is not in the registry.
        """
        if oblock_id in self.backend.ob_table:
            return
        oblock = None if self.registry is None else self.registry.oblock(oblock_id)
        if oblock is None:
            where = "the observation results" if self.registry is None else "the observation results or the registry"
            raise ValueError(f"oblock_id={oblock_id} is not in {where}")
        for child in oblock.get("children", []):
            self.restore_oblock(child)
        _logger.debug("restore oblock_id=%s from the registry", oblock_id)
        self.backend.add_obs([oblock])

    def new_task(self, request, request_params):
        """Create a new task

        If there is a registry, the task is recorded in it,
        and the id of the task is assigned by the registry.
        """
        task = self.backend.new_task(request, request_params)
        if self.registry is not None:
            oblock_id = request_params.get("oblock_id")
            oblock = self.backend.ob_table.get(oblock_id)
            task.id = self.registry.new_task(task, oblock)
        return task

    def store_task(self, task):
        """Store a task and its result.

        Write the products of the result, the result file and the task file in
        the results directory of the task, and update the task and its result in
        the DAL and, if there is one, in the registry.
        """

        result_dir_rel = task.request_runinfo["results_dir"]
        result_dir = os.path.join(self.basedir, result_dir_rel)

        obsid = task.request_params["oblock_id"]
        result_file = fill_template(self.resultfile_tmpl, obsid=obsid, taskid=task.id)
        task_file = fill_template(self.taskfile_tmpl, obsid=obsid, taskid=task.id)

        with working_directory(result_dir):

            task_repr = task.__dict__.copy()

            # save to disk the RecipeResult part and return the file to save it
            if task.result is not None:
                uuid = task.result.uuid
                qc = task.result.qc
                _logger.info("storing result uuid=%s, quality=%s", uuid, qc)
                if qc != QC.GOOD:
                    for key, val in task.result.stored().items():
                        val = getattr(task.result, key)
                        if hasattr(val, "quality_control"):
                            _logger.info(
                                "with field %s=%s, quality=%s",
                                key,
                                val,
                                val.quality_control,
                            )

                result_repr = self.store_result_to(task.result, obsid=obsid, taskid=task.id)
                # Change result structure by filename
                task_repr["result"] = result_file

                with open(result_file, "w+") as fd:
                    self.serializer(result_repr, fd)

            tid = task.id
            _logger.info("storing task id=%s", tid)
            with open(task_file, "w+") as fd:
                self.serializer(task_repr, fd)

        self.backend.update_task(task)
        if task.result is not None:
            self.backend.update_result(task, result_repr, result_file)

        if self.registry is not None:
            self.registry.update_task(task)
            if task.result is not None:
                self.registry.new_result(task, result_repr, result_file)

    def create_workenv(self, task):
        """Return the ``WorkEnvironment`` of a task, with the directory names of the templates."""

        obsid = task.request_params["oblock_id"]
        work_dir = fill_template(self.workdir_tmpl, obsid=obsid, taskid=task.id)
        result_dir = fill_template(self.resultdir_tmpl, obsid=obsid, taskid=task.id)

        workenv = WorkEnvironment(self.datadir, self.basedir, work_dir, result_dir)

        return workenv


class ProcessingTask:
    """Reduction task, with its request, its run information, its result and its state.

    The state is 0 (created), 1 (running), 2 (finished) or 3 (failed).
    """

    def __init__(self):

        self.result = None
        self.id = 1

        self.time_create = datetime.datetime.now(datetime.UTC)
        # None until the recipe runs
        self.time_start = None
        self.time_end = None
        self.request = "reduce"
        self.request_params = {}
        self.request_runinfo = self._init_runinfo()
        self.state = 0

    @classmethod
    def _init_runinfo(cls):
        """Return the initial run information, with unknown runner and runner version."""
        request_runinfo = dict(runner="unknown", runner_version="unknown")
        return request_runinfo


class WorkEnvironment:
    """Data, work and results directories of a reduction.

    Parameters
    ----------
    datadir : str
        Directory of the raw data.
    basedir : str
        Base directory of the work and results directories.
    workdir : str
        Work directory, relative to ``basedir``.
    resultsdir : str
        Results directory, relative to ``basedir``.
    """

    def __init__(self, datadir, basedir, workdir, resultsdir):

        self.basedir = basedir

        self.workdir_rel = workdir
        self.workdir = os.path.abspath(os.path.join(self.basedir, workdir))

        self.resultsdir_rel = resultsdir
        self.resultsdir = os.path.abspath(os.path.join(self.basedir, resultsdir))

        self.datadir_rel = datadir
        self.datadir = os.path.abspath(datadir)

        # The source of each file installed in workdir
        self._installed = {}

    def sane_work(self):
        """Create the work and results directories if they do not exist."""
        _logger.debug("check workdir for working: %r", self.workdir_rel)
        make_sure_path_exists(self.workdir)

        _logger.debug("check resultsdir to store results %r", self.resultsdir_rel)
        make_sure_path_exists(self.resultsdir)

    def installfiles_stage1(self, obsres, action="copy"):
        """Install the frames of the observation result in the work directory.

        Each frame is copied or linked (``action`` is ``copy`` or ``link``) to the
        work directory, and its filename is replaced by the destination path.
        Relative filenames are relative to the data directory. Frames with the
        same name get the UUID of their header in the name. Return ``obsres``.
        """
        import astropy.io.fits as fits

        install_if_needed = self._calc_install_if_needed(action)

        _logger.debug("installing files from observation result")
        tails = []
        sources = []
        for f in obsres.images:
            if not os.path.isabs(f.filename):
                complete = os.path.abspath(os.path.join(self.datadir, f.filename))
            else:
                complete = f.filename
            head, tail = os.path.split(complete)
            # initial.append(complete)
            tails.append(tail)
            #            heads.append(head)
            sources.append(complete)

        dupes = self.check_duplicates(tails)

        for src, obj in zip(sources, obsres.images):
            head, tail = os.path.split(src)
            if tail in dupes:
                # extract UUID
                hdr = fits.getheader(src)
                img_uuid = hdr["UUID"]
                root, ext = os.path.splitext(tail)
                key = f"{root}_{img_uuid}{ext}"

            else:
                key = tail
            dest = self._destination(src, key)
            # Update filename in DataFrame
            obj.filename = dest
            install_if_needed(key, src, dest)

        if obsres.results:
            _logger.warning("not installing files in 'results")
        return obsres

    def _calc_install_if_needed(self, action):
        """Return the method that installs a file with ``action`` (``copy`` or ``link``)."""
        if action not in ["copy", "link"]:  # , 'symlink', 'hardlink']:
            raise ValueError(f"{action} action is not allowed")

        _logger.debug(f'installing files with "{action}"')

        if action == "copy":
            install_if_needed = self.copy_if_needed
        elif action == "link":
            install_if_needed = self.link_if_needed
        else:
            raise ValueError(f"{action} action is not allowed")
        return install_if_needed

    def installfiles_stage2(self, reqs, action="copy"):
        """Install the frames of the recipe input ``reqs`` in the work directory.

        Each frame is copied or linked (``action`` is ``copy`` or ``link``) from
        the data directory, and its filename is replaced by the name of the
        installed file.
        """
        _logger.debug("installing files from requirements")

        install_if_needed = self._calc_install_if_needed(action)

        for _, req in reqs.stored().items():
            if isinstance(req.type, DataFrameType):
                value = getattr(reqs, req.dest)
                if value is None:
                    continue

                complete = os.path.abspath(os.path.join(self.datadir, value.filename))

                head, tail = os.path.split(value.filename)
                dest = self._destination(complete, tail)

                install_if_needed(value.filename, complete, dest)
                # The recipe runs in the work directory, where the file is installed,
                # as with the frames in adapt_obsres
                value.filename = os.path.basename(dest)

    # Used by numina-plugin-db
    def copyfiles_stage1(self, obsres):
        """Copy the frames of the observation result to the work directory."""
        return self.installfiles_stage1(obsres, action="copy")

    def check_duplicates(self, tails):
        """Return the set of names that appear more than once in ``tails``."""
        seen = set()
        dupes = set()
        for tail in tails:
            if tail in seen:
                dupes.add(tail)
            else:
                seen.add(tail)
        return dupes

    # Used by numina-plugin-db
    def copyfiles_stage2(self, reqs):
        """Copy the frames of the recipe input to the work directory."""
        return self.installfiles_stage2(reqs, action="copy")

    def _destination(self, src, name):
        """The path of `src` in workdir, with the name `name`.

        If another file was installed with the same name, a part of the hash
        of the path of `src` is added to the name, so that the name does not
        change between runs.
        """
        src_real = os.path.realpath(src)
        dest = os.path.join(self.workdir, name)
        installed = self._installed.get(dest)
        if installed is not None and installed != src_real:
            root, ext = os.path.splitext(name)
            digest = hashlib.sha1(src_real.encode()).hexdigest()[:8]
            dest = os.path.join(self.workdir, f"{root}_{digest}{ext}")
            _logger.debug("%r is already installed from other file, using %r", name, dest)
        self._installed[dest] = src_real
        return dest

    def copy_if_needed(self, key, src, dest):
        """Copy `src` to `dest`, unless `dest` is already a copy of `src`.

        `dest` is a copy if it is a regular file with the size and the
        modification time of `src`, as in the quick check of rsync. The copy
        keeps the modification time of `src`. `key` is used in the log.
        """
        if is_copy_of(src, dest):
            _logger.debug("copying %r not needed", key)
            return

        if os.path.islink(dest):
            # A link from a previous run, it may point to src
            os.remove(dest)
        _logger.debug("copying %r to %r", key, self.workdir)
        shutil.copy2(src, dest)

    def link_if_needed(self, key, src, dest):
        """Create ``dest`` as a symbolic link to ``src``, replacing ``dest`` if it exists.

        ``key`` is used in the log.
        """
        _logger.debug("linking %r to %r", key, self.workdir)
        try:
            # Remove destination
            os.remove(dest)
        except OSError:
            pass
        os.symlink(src, dest)

    def adapt_obsres(self, obsres):
        """Adapt obsres after file copy"""

        _logger.debug("adapt observation result for work dir")
        for f in obsres.images:
            # Remove path components
            f.filename = os.path.basename(f.filename)
        return obsres


def is_copy_of(src, dest, modify_window=MODIFY_WINDOW):
    """Check if `dest` is a copy of `src`.

    It is if `dest` is a regular file, not a link, with the size of `src`
    and a modification time that differs at most `modify_window` seconds.
    """
    if os.path.islink(dest) or not os.path.isfile(dest):
        return False
    src_stat = os.stat(src)
    dest_stat = os.stat(dest)
    return src_stat.st_size == dest_stat.st_size and abs(src_stat.st_mtime - dest_stat.st_mtime) <= modify_window


def make_sure_path_exists(path):
    """Create the directory ``path`` and its parents, if it does not exist."""
    try:
        os.makedirs(path)
    except (OSError, IOError) as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_com_store(sys_drps, profile_path_extra=None):
    """Return the store of instrument profiles of the DRPs, and of the directory ``profile_path_extra``."""
    import numina.instrument.assembly as asbl

    com_store = asbl.load_panoply_store(sys_drps, profile_path_extra)
    return com_store


def deep_merge(initial_schema, loaded_data):
    """Merge the control data ``loaded_data`` into ``initial_schema`` and return it.

    The keys other than ``requirements`` replace those of ``initial_schema``.
    The entries of ``requirements`` (by instrument, profile, pipeline and
    mode) are merged: an entry with the name and tags of an existing entry
    replaces its content, the others are appended. ``initial_schema`` is
    modified in place.
    """
    # copy all keys except requirements
    for key in loaded_data:
        if key == "requirements":
            reqs = loaded_data[key]
            init_reqs = initial_schema[key]
            for ins_name in reqs:
                for epoch in reqs[ins_name]:
                    for pipe in reqs[ins_name][epoch]:
                        for recipe in reqs[ins_name][epoch][pipe]:
                            init_recipe = (
                                init_reqs.setdefault(ins_name, {})
                                .setdefault(epoch, {})
                                .setdefault(pipe, {})
                                .setdefault(recipe, [])
                            )
                            for entry in reqs[ins_name][epoch][pipe][recipe]:
                                entry_name = entry["name"]
                                entry_tags = entry["tags"]
                                for init_entry in init_recipe:
                                    if init_entry["name"] == entry_name and init_entry["tags"] == entry_tags:
                                        init_entry["content"] = entry["content"]
                                        break
                                else:
                                    init_recipe.append(entry)
        else:
            initial_schema[key] = loaded_data[key]
    return initial_schema


def add_missing_entries(entries, defaults):
    """Append to entries the defaults with a name and tags not present in entries"""
    for entry in defaults:
        for current in entries:
            if current["name"] == entry["name"] and current["tags"] == entry["tags"]:
                break
        else:
            entries.append(entry)
    return entries


def process_format_version_1(
    sys_drps,
    components,
    basedir: str,
    rootdir: str,
    loaded_data,
    loaded_data_extra=None,
    resultdir_tmpl="obsid{obsid}_results",
    resultfile_tmpl="result.json",
) -> HybridDAL:
    """Return the DAL of a control file in format 1.

    Parameters
    ----------
    sys_drps : numina.drps.drpsystem.DrpSystem
        The DRPs.
    components : dict
        Store of instrument profiles.
    basedir : str
        Base directory of the reductions.
    rootdir : str
        Directory of calibrations.
    loaded_data : dict
        Contents of the control file.
    loaded_data_extra : dict, optional
        Extra values of the control, from the command line.
    resultdir_tmpl, resultfile_tmpl : str
        Templates of the names of the results directory and the result file,
        used to find the results of other observing blocks.
    """
    backend = HybridDAL(
        sys_drps,
        rootdir,
        [],
        loaded_data,
        extra_data=loaded_data_extra,
        basedir=basedir,
        components=components,
        resultdir_tmpl=resultdir_tmpl,
        resultfile_tmpl=resultfile_tmpl,
    )
    return backend


def run_templates(section, db_section=None):
    """Templates of the names of directories and files

    The templates are read from `section` ([tool.run]). If `db_section`
    ([tool.db]) is not None, the templates defined there replace them.
    """
    names = ["workdir_tmpl", "resultdir_tmpl", "resultfile_tmpl", "taskfile_tmpl"]
    templates = {name: section[name] for name in names}
    if db_section is not None:
        for name in names:
            if name in db_section:
                templates[name] = db_section[name]
    # the names of the products are the same with or without registry
    templates["product_tmpl"] = section.get("product_tmpl", "{name}")
    return templates


def create_datamanager(config, reqfile, extra_control=None, profile_path_extra=None) -> DataManager:
    """Create the ``DataManager`` of a reduction from the configuration and the control file.

    The default requirements of the DRPs are merged with those of the control
    file. The directories and the templates of names are read from
    ``[tool.run]`` (and ``[tool.db]``), and the registry of reductions is
    opened if ``[tool.db]`` has ``file``.

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration of numina, with the section ``[tool.run]``.
    reqfile : str or None
        Control file, in format 1.
    extra_control : dict, optional
        Extra values of the control, as strings that are parsed as YAML.
    profile_path_extra : str, optional
        Directory with additional instrument configurations.

    Raises
    ------
    ValueError
        If the format of the control file is not 1, or the control file
        is not valid.
    """

    # This should go before we load CL file
    # load additional reduction defaults
    sys_drps = numina.drps.get_system_drps()
    initial_schema = {"version": 1, "requirements": {}}
    for ins, drp in sys_drps.query_all().items():
        instrument_name = drp.name
        values = drp.default_requirements()
        # insert requirements
        reqs = values.get("requirements", {})
        reqs_ins = reqs.get(instrument_name, {})
        if reqs_ins:
            initial_schema["requirements"][instrument_name] = reqs_ins

    section = config["tool.run"]
    # The work and result directories are created inside basedir,
    # and relative paths are resolved against it
    basedir = os.path.abspath(section["basedir"])
    datadir = os.path.join(basedir, section["datadir"])
    calibsdir = section.get("calibsdir")

    if reqfile:
        _logger.info("reading control from %s", reqfile)
        with open(reqfile, "r") as fd:
            # an empty file is an empty control
            loaded_data = yaml.safe_load(fd) or {}
    else:
        _logger.info("no control file")
        loaded_data = {}

    if extra_control:
        _logger.info("extra control %s", extra_control)
        loaded_data_extra = parse_as_yaml(extra_control)
    else:
        loaded_data_extra = None

    control_format = loaded_data.get("version", 1)
    _logger.info("control format version %d", control_format)
    if control_format != 1:
        msg = f"format {control_format} of the control file {reqfile} is not supported, only format 1"
        if control_format == 2:
            msg += ", the reductions can be recorded in a registry (numina run --db FILE)"
        raise ValueError(msg)
    if reqfile:
        validate_schema(loaded_data, "control", source=reqfile)

    components = create_com_store(sys_drps, profile_path_extra)
    # What rootdir are going to use
    if calibsdir is None:
        _logger.debug("loading rootdir from control file data")
        calibsdir = loaded_data.get("rootdir", None)

    if calibsdir is not None:
        _logger.debug(f"current calibsdir is '{calibsdir}'")
        if not os.path.isabs(calibsdir):
            calibsdir = os.path.join(basedir, calibsdir)
        calibsdir = os.path.normpath(calibsdir)
    else:
        calibsdir = ""

    db_section = config["tool.db"] if config.has_section("tool.db") else {}
    use_registry = bool(db_section.get("file"))
    templates = run_templates(section, db_section if use_registry else None)

    merged_data = deep_merge(initial_schema, loaded_data)
    # The backend uses the same templates to find the results of other OBs
    _backend = process_format_version_1(
        sys_drps,
        components,
        basedir,
        calibsdir,
        merged_data,
        loaded_data_extra,
        resultdir_tmpl=templates["resultdir_tmpl"],
        resultfile_tmpl=templates["resultfile_tmpl"],
    )
    datamanager = DataManager(basedir, datadir, _backend)
    datamanager.set_templates(
        templates["workdir_tmpl"],
        templates["resultdir_tmpl"],
        templates["resultfile_tmpl"],
        templates["taskfile_tmpl"],
        templates["product_tmpl"],
    )

    if use_registry:
        dbpath = os.path.join(basedir, db_section["file"])
        _logger.info("registry of reductions in %s", dbpath)
        datamanager.registry = Registry(dbpath, basedir=basedir)
        # The backend reads the products and results from the registry
        datamanager.backend.registry = datamanager.registry
        if "{taskid}" not in datamanager.resultdir_tmpl:
            _logger.warning(
                "resultdir_tmpl=%s does not use {taskid}, the results of a new reduction "
                "of an OB replace the results of the previous ones in the registry",
                datamanager.resultdir_tmpl,
            )

    # This should go before we load CL file
    # load additional reduction defaults
    for ins, drp in datamanager.backend.drps.query_all().items():

        values = drp.default_requirements()
        # insert requirements
        reqs = values.get("requirements", {})
        reqs_ins = reqs.get(ins, {})
        for prof_name in reqs_ins:
            node_pln = reqs_ins[prof_name]
            for pln_name in node_pln:
                node_obs = node_pln[pln_name]
                for obs_name in node_obs:
                    params = node_obs[obs_name]
                    # Set instrument if not defined
                    n1 = datamanager.backend.req_table.setdefault(ins, {})
                    # Set instrumental profile if not defined
                    n2 = n1.setdefault(prof_name, {})
                    # Set pipeline if not defined
                    n3 = n2.setdefault(pln_name, {})
                    # Insert params, without replacing the values
                    # already loaded from the control file
                    n4 = n3.setdefault(obs_name, [])
                    add_missing_entries(n4, params)

    return datamanager


def load_observations(obfiles, is_session=False):
    """Observing mode processing mode of numina."""

    # Loading observation result if exists
    loaded_obs = []
    sessions = []

    for obfile in obfiles:
        with open(obfile) as fd:
            if is_session:
                _logger.info("session file from %r", obfile)
                sess = yaml.safe_load(fd)
                sessions.append(sess["session"])
            else:
                _logger.info("observation results from %r", obfile)
                sess = []
                for idx, doc in enumerate(yaml.safe_load_all(fd), start=1):
                    if doc is None:
                        # an empty document, for example after a final '---'
                        continue
                    validate_schema(doc, "oblock", source=f"{obfile} (document {idx})")
                    enabled = doc.get("enabled", True)
                    docid = doc["id"]
                    # the requirements of the OB are read from the OB table
                    sess.append(dict(id=docid, enabled=enabled))
                    if enabled:
                        _logger.debug("load observation result with id %s", docid)
                    else:
                        _logger.debug("skip observation result with id %s", docid)

                    loaded_obs.append(doc)

                sessions.append(sess)
    return sessions, loaded_obs


class ReductionRequests:
    """Reduction requests, how to reduce the observing blocks

    The requests are read from a YAML file, with one request in each
    document (see ``request-schema.json``). A request with ``id``
    applies to that observing block, a request without ``id`` to all
    the others.

    Parameters
    ----------
    requests : dict, optional
        Requests by id of the observing block.
    default : dict, optional
        Request of the blocks without their own request.
    """

    #: The keys of a request and the arguments of run_reduce
    ARGUMENTS = {"requirements": "requirements", "pipeline": "pipeline", "profile": "profile", "mode": "as_mode"}

    def __init__(self, requests=None, default=None):
        #: Requests by id of the observing block
        self.requests = {} if requests is None else requests
        #: Request of the blocks without their own request
        self.default = default

    @classmethod
    def load(cls, filename):
        """Read the requests of a YAML file

        Raises
        ------
        numina.schemas.SchemaValidationError
            If a request is not valid.
        ValueError
            If there are two requests for the same block, or two without id.
        """
        requests = {}
        default = None
        with open(filename) as fd:
            for idx, doc in enumerate(yaml.safe_load_all(fd), start=1):
                if doc is None:
                    continue
                validate_schema(doc, "request", source=f"{filename} (document {idx})")
                if "id" in doc:
                    if doc["id"] in requests:
                        raise ValueError(f"{filename}: two requests for oblock_id={doc['id']}")
                    requests[doc["id"]] = doc
                elif default is not None:
                    raise ValueError(f"{filename}: two requests without id")
                else:
                    default = doc
        return cls(requests, default)

    def ids(self):
        """The ids of the blocks with their own request, in the order of the file"""
        return list(self.requests)

    def request(self, oblock_id):
        """The request of the block `oblock_id`, the default if it has none

        The ids are compared as strings too, an id can be an integer in a
        file and a string in the command line.
        """
        if oblock_id in self.requests:
            return self.requests[oblock_id]
        for key, request in self.requests.items():
            if str(key) == str(oblock_id):
                return request
        return self.default

    def arguments(self, oblock_id):
        """The keyword arguments of run_reduce for the block `oblock_id`"""
        request = self.request(oblock_id) or {}
        return {self.ARGUMENTS[key]: value for key, value in request.items() if key in self.ARGUMENTS}


def parse_as_yaml(strdict):
    """Parse a dictionary of strings as if yaml reads it"""
    interm = ""
    for key, val in strdict.items():
        interm = f"{key}: {val}, {interm}"
    fin = "{%s}" % interm

    return yaml.safe_load(fin)
