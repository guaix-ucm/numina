#
# Copyright 2008-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import contextlib
import datetime
import logging
import os

import numina.exceptions
from numina.user.logconf import LOGCONF

from numina.util.context import working_directory
from numina.util.fqn import fully_qualified_name
from numina.core.recipeinout import check_product_tags
from numina.user.helpers import ProcessingTask, DataManager

_logger = logging.getLogger(__name__)


def describe_value(value):
    """A short description of the value of a requirement, that can be stored in JSON"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, (bool, int, float, str)) for v in value):
        return list(value)
    filename = getattr(value, "filename", None)
    if filename is not None:
        return filename
    uuid = getattr(value, "uuid", None)
    if uuid is not None:
        return f"{type(value).__name__} uuid={uuid}"
    return type(value).__name__


def describe_inputs(recipe, rinput, dal, request_requirements, ob_requirements):
    """Where the value of each requirement of the recipe comes from, and the value

    The values found by the DAL are recorded in ``dal.sources``. The others
    come from the reduction request, the observing block, the results of
    other blocks (ResultOf) or the default value.
    """
    from numina.core.query import ResultOf
    import numina.types.obsresult as obtype

    sources = getattr(dal, "sources", {})
    inputs = {}
    for key, req in recipe.requirements().items():
        if isinstance(req.type, obtype.ObservationResultType):
            continue
        names = {req.dest, getattr(req, "alias", None)}
        found = [name for name in names if name in sources]
        if found:
            info = dict(sources[found[0]])
        elif names & set(request_requirements):
            info = {"source": "request"}
        elif names & set(ob_requirements):
            info = {"source": "observation block"}
        elif isinstance(recipe.query_options.get(key), ResultOf):
            info = {"source": "result of"}
        else:
            info = {"source": "default"}
        info["value"] = describe_value(getattr(rinput, key))
        inputs[key] = info
    return inputs


def requirement_types(recipe, drp):
    """The names of the product types of each requirement of the recipe, by name and alias"""
    types = {}
    for req in recipe.requirements().values():
        node_types = getattr(req.type, "node_type", None) or [req.type]
        labels = set()
        for node_type in node_types:
            if node_type.isproduct():
                label = drp.product_label(node_type)
                labels.add(label[:-2] if label.endswith("()") else label)
        for name in (req.dest, getattr(req, "alias", None)):
            if name is not None:
                types[name] = labels
    return types


def resolve_references(obsres, datastore, request_requirements, recipe):
    """Replace the references to the registry in the requirements of `obsres` by their values

    The product types are checked against those of the requirements of the
    recipe. The source of each value is recorded in the DAL.

    Raises
    ------
    numina.exceptions.ValidationError
        If a reference is not valid, its type is not that of the
        requirement, or there is no registry.
    """
    from numina.dal.registry import is_reference

    requirements = obsres.requirements
    resolved = {}
    if not any(is_reference(value) for value in requirements.values()):
        return resolved
    drp = datastore.backend.drps.query_by_name(obsres.instrument)
    types = requirement_types(recipe, drp)
    for key, value in requirements.items():
        if not is_reference(value):
            continue
        if datastore.registry is None:
            msg = f"requirement {key} is a reference to the registry, but there is no registry of reductions"
            raise numina.exceptions.ValidationError(msg)
        # an unknown requirement, or one that is not a product, is not checked
        accepted = types.get(key) or None
        try:
            resolved_value, details = datastore.registry.resolve_reference(value, key, types=accepted)
        except (ValueError, OSError) as error:
            raise numina.exceptions.ValidationError(f"requirement {key}: {error}") from error
        requirements[key] = resolved_value
        requested_in = "request" if key in request_requirements else "observation block"
        datastore.backend.record_source(key, "registry", reference=value, requested_in=requested_in, **details)
        if "tags" in details:
            if "product_id" in details:
                what = f"the product {details['product_id']}"
            else:
                what = f"the field {value.get('field', key)} of the result {details['result_id']}"
            resolved[key] = (what, details["tags"])
    return resolved


def check_reference_tags(recipe, obsres, resolved):
    """Check that the tags of the products of the references match those of obsres

    `resolved` are the references resolved by resolve_references. The tags
    are checked with the query expression of the type of the requirement,
    as in the automatic selection of products.

    Raises
    ------
    numina.exceptions.ValidationError
        If the tags of a product do not match.
    """
    by_name = {}
    for req in recipe.requirements().values():
        for name in (req.dest, getattr(req, "alias", None)):
            if name is not None:
                by_name[name] = req
    for key, (what, tags) in resolved.items():
        req = by_name.get(key)
        if req is not None:
            check_product_tags(key, what, req.type, tags, obsres.tags)


def run_reduce(
    datastore: DataManager,
    obsid,
    as_mode=None,
    pipeline=None,
    profile=None,
    requirements=None,
    copy_files=False,
    validate_inputs=False,
    validate_results=False,
    strict_inputs=False,
) -> ProcessingTask:
    """Observing mode processing mode of numina.

    If pipeline is None, the pipeline of the observation result is used.
    If profile is None, the instrument configuration is selected
    from the images of the observation result.
    """

    # The block can be in the registry of reductions
    datastore.restore_oblock(obsid)

    request = "reduce"
    request_params = dict()

    request_params["oblock_id"] = obsid
    request_params["pipeline"] = pipeline
    request_params["instrument_configuration"] = profile
    request_params["intermediate_results"] = True
    request_params["validate_results"] = validate_results
    request_params["validate_inputs"] = validate_inputs
    request_params["strict_inputs"] = strict_inputs
    request_params["copy_files"] = copy_files
    request_params["mode"] = as_mode

    requirements = {} if requirements is None else requirements
    request_params["requirements"] = requirements

    formater_extended = LOGCONF["formatters"]["extended"]

    # If we put the name of logging it in LOGCONF, the file is created when
    # logging system is initialized, we don't want that
    # handler_file = LOGCONF['handlers']['processing_file']['filename']
    handler_file = "processing.log"
    logger_control = dict(
        default=__name__, root_levels=["numina"], logfile=handler_file, format=formater_extended["format"], enabled=True
    )
    request_params["logger_control"] = logger_control

    task = datastore.new_task(request, request_params)

    try:
        completed_task = run_task_reduce(task, datastore)
    except Exception:
        # The task can fail before running the recipe
        task.state = 3
        try:
            datastore.store_task(task)
        except Exception:
            _logger.exception("error storing failed task id=%s", task.id)
        raise
    datastore.store_task(completed_task)
    return completed_task


def run_task_reduce(task: ProcessingTask, datastore: DataManager) -> ProcessingTask:
    """Run the recipe of a reduction task.

    Load the observing block of the task, select its recipe, build the recipe
    input through the DAL of ``datastore``, install the files in the work
    directory and run the recipe there, logging to ``processing.log`` in the
    results directory. The run information is recorded in the task. The task
    is not stored, ``run_reduce`` does that.

    Returns
    -------
    ProcessingTask
        The task, with the result of the recipe and its final state.

    Raises
    ------
    ValueError
        If the recipe input cannot be built, or if a requirement of the
        observing block is not used by the recipe and ``strict_inputs`` is set.
    numina.exceptions.ValidationError
        If the recipe input is not valid.
    """

    obsid = task.request_params["oblock_id"]
    request_profile = task.request_params["instrument_configuration"]
    as_mode = task.request_params["mode"]

    _logger.info(f"processing OB with id={obsid}")

    workenv = datastore.create_workenv(task)
    task.request_runinfo["results_dir"] = workenv.resultsdir_rel
    task.request_runinfo["work_dir"] = workenv.workdir_rel
    workenv.sane_work()

    # Roll back to cwd after leaving the context
    with working_directory(workenv.datadir):

        # here the configuration object has been updated
        # and configured with one image
        # if we have a ResultOf, ObservingMode.build_ob will insert results
        obsres = datastore.backend.obsres_from_oblock_id(obsid, as_mode=as_mode, profile=request_profile)
        _logger.debug("instrument profile is %s", obsres.profile)
        task.request_params["instrument_configuration"] = obsres.profile
        # The requirements of the OB, before the request overrides them
        ob_requirements = dict(obsres.requirements)
        # Merge requirements passed from above
        obsres.requirements.update(task.request_params["requirements"])
        # The pipeline of the request overrides the one in the OB
        if task.request_params["pipeline"] is not None:
            obsres.pipeline = task.request_params["pipeline"]
        task.request_params["pipeline"] = obsres.pipeline
        _logger.debug("pipeline is %s", obsres.pipeline)

        recipe = datastore.backend.search_recipe_from_ob(obsres)
        _logger.debug("recipe class is %s", recipe.__class__)

        # The DAL records where it finds the values
        datastore.backend.sources = {}
        # The references are resolved with the types of the requirements of the recipe
        references = resolve_references(obsres, datastore, task.request_params["requirements"], recipe)

        recipe.intermediate_results = task.request_params["intermediate_results"]
        recipe.validate_inputs = task.request_params["validate_inputs"]
        recipe.validate_results = task.request_params["validate_results"]

        # Update runinfo
        recipe.runinfo["runner"] = task.request_runinfo["runner"]
        recipe.runinfo["runner_version"] = task.request_runinfo["runner_version"]
        recipe.runinfo["task_id"] = task.id
        recipe.runinfo["data_dir"] = workenv.datadir
        recipe.runinfo["work_dir"] = workenv.workdir
        recipe.runinfo["results_dir"] = workenv.resultsdir
        recipe.runinfo["intermediate_results"] = task.request_params["intermediate_results"]

        _logger.debug("recipe created")

        try:
            # here the tags have been updated
            rinput = recipe.build_recipe_input(obsres, datastore.backend)
            # The tags of obsres are extracted in build_recipe_input
            check_reference_tags(recipe, obsres, references)
        except (ValueError, numina.exceptions.ValidationError) as err:
            _logger.error("During recipe input construction")
            _logger.error("%s", err)
            raise

        _logger.debug("recipe input created")
        # The configuration is selected again if the inputs include
        # the results of other OBs (ResultOf)
        task.request_params["instrument_configuration"] = obsres.profile
        # Show the actual inputs
        for key, val in obsres.requirements.items():
            if key not in recipe.requirements():
                msg = f'"{key}: {val}" present in OB requirements, but not used'
                if task.request_params["strict_inputs"]:
                    _logger.error(msg)
                    raise ValueError(msg)
                else:
                    _logger.warning(msg)

        for key, req in recipe.requirements().items():
            v = getattr(rinput, key)
            _logger.debug("recipe requires %r, value is %s", key, v)

        task.request_runinfo["inputs"] = describe_inputs(
            recipe, rinput, datastore.backend, task.request_params["requirements"], ob_requirements
        )

        for req in recipe.products().values():
            _logger.debug("recipe provides %s, %s", req.type.__class__.__name__, req.description)

    # Load recipe control and recipe parameters from file
    task.request_runinfo["instrument"] = obsres.instrument
    task.request_runinfo["pipeline"] = obsres.pipeline
    task.request_runinfo["mode"] = obsres.mode
    task.request_runinfo["recipe_class"] = recipe.__class__.__name__
    task.request_runinfo["recipe_fqn"] = fully_qualified_name(recipe.__class__)
    task.request_runinfo["recipe_version"] = recipe.__version__
    root_level_logger = task.request_runinfo["recipe_fqn"].split(".")[0]

    # Copy files
    if task.request_params["copy_files"]:
        install_action = "copy"
        _logger.debug("copy files to work directory")
    else:
        install_action = "link"
        _logger.debug("link files to work directory")

    workenv.installfiles_stage1(obsres, action=install_action)
    workenv.installfiles_stage2(rinput, action=install_action)
    workenv.adapt_obsres(obsres)

    logger_control = task.request_params["logger_control"]
    # configure recipe root level
    with config_recipe_logger(root_level_logger):
        logger_control["root_levels"].append(root_level_logger)
        # Add file logging
        with logger_manager(logger_control, workenv.resultsdir):
            with working_directory(workenv.workdir):
                completed_task = run_recipe_timed(task, recipe, rinput)

    # datastore.store_task(completed_task)
    return completed_task


def run_recipe_timed(task: ProcessingTask, recipe, rinput) -> ProcessingTask:
    """Run the recipe and count the time it takes."""
    _logger.info("running recipe")
    task.state = 1
    task.time_start = datetime.datetime.now()
    #
    try:
        task.result = recipe(rinput)
        task.state = 2
    except Exception as error:
        task.state = 3
        # In the log of the reduction, the traceback only with DEBUG,
        # the console shows it when the error is raised
        _logger.error("the recipe failed, %s: %s", type(error).__name__, error)
        _logger.debug("traceback of the error of the recipe", exc_info=True)
        raise
    finally:
        task.time_end = datetime.datetime.now()
    return task


@contextlib.contextmanager
def logger_manager(logger_control, result_dir):
    """Add a FileHandler to existing loggers

    We add a FileHandler to the loggers named in
    logger_control['root_levels'] if
    logger_control['root_levels'] is True
    with level DEBUG

    The warnings issued in the context are captured, and logged
    in the file by the logger 'py.warnings'.

    After exiting the context, the handlers are removed and the
    levels of the loggers restored
    """
    # Creating custom logger file
    if logger_control["root_levels"]:
        logfile = os.path.join(result_dir, logger_control["logfile"])
        logformat = logger_control["format"]
        recipe_formatter = logging.Formatter(logformat)
        fh = logging.FileHandler(logfile, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(recipe_formatter)
    else:
        fh = logging.NullHandler()

    root_levels = logger_control["root_levels"]
    recipe_loggers = []
    saved_levels = []
    for lname in root_levels:
        recipe_logger = logging.getLogger(lname)
        saved_levels.append(recipe_logger.level)
        recipe_logger.setLevel(logging.DEBUG)
        recipe_logger.addHandler(fh)
        recipe_loggers.append(recipe_logger)
    # The warnings, logged in py.warnings. The logger is enabled, a
    # previous logging.config.dictConfig disables it if it existed
    warnings_logger = logging.getLogger("py.warnings")
    warnings_disabled = warnings_logger.disabled
    warnings_logger.disabled = False
    warnings_logger.addHandler(fh)
    capturing = getattr(logging, "_warnings_showwarning", None) is not None
    logging.captureWarnings(True)
    try:
        yield recipe_loggers
    finally:
        if not capturing:
            logging.captureWarnings(False)
        warnings_logger.removeHandler(fh)
        warnings_logger.disabled = warnings_disabled
        for recipe_logger, level in zip(recipe_loggers, saved_levels):
            recipe_logger.removeHandler(fh)
            recipe_logger.setLevel(level)
        # Close the log file
        fh.close()


@contextlib.contextmanager
def config_recipe_logger(root_level_logger, ref_logger="numina"):
    """Configure root level logger for recipe.

    The recipe formatter is 'detailed'
    The logger is set to DEBUG (this is needed by the FileHandler later
    The StreamHandler is set to the same level as the logger of numina

    """
    numina_logger = logging.getLogger(ref_logger)
    recipe_logger = logging.getLogger(root_level_logger)
    # Save state of logger
    save_level = recipe_logger.getEffectiveLevel()
    recipe_logger.setLevel(logging.DEBUG)
    save_propagate = recipe_logger.propagate
    recipe_logger.propagate = False

    # create formatter
    formater_dd = LOGCONF["formatters"]["detailed"]
    detailed_formatter = logging.Formatter(fmt=formater_dd.get("format"))
    # create handler, ignoring configuration here
    # handerl_dd = numina_cli_logconf['handlers']['detailed_console']
    # handerl_dd_level =
    sh = logging.StreamHandler()
    sh.setLevel(numina_logger.getEffectiveLevel())
    sh.setFormatter(detailed_formatter)
    recipe_logger.addHandler(sh)

    try:
        yield recipe_logger
    finally:
        # Restore state
        recipe_logger.removeHandler(sh)
        recipe_logger.propagate = save_propagate
        recipe_logger.setLevel(save_level)
