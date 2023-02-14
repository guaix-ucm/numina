#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import contextlib
import datetime
import logging
import os



import numina.exceptions
from numina.user.logconf import LOGCONF
from numina.util.fqn import fully_qualified_name
from numina.util.context import working_directory


_logger = logging.getLogger(__name__)


def run_reduce(datastore, obsid, as_mode=None, requirements=None, copy_files=False,
               validate_inputs=False, validate_results=False):
    """Observing mode processing mode of numina."""

    request = 'reduce'
    request_params = dict()

    request_params['oblock_id'] = obsid
    request_params["pipeline"] = 'default'  # args.pipe_name
    request_params["instrument_configuration"] = 'default'  # args.insconf
    request_params["intermediate_results"] = True
    request_params["validate_results"] = validate_results
    request_params["validate_inputs"] = validate_inputs
    request_params["copy_files"] = copy_files
    request_params["mode"] = as_mode

    requirements = {} if requirements is None else requirements
    request_params["requirements"] = requirements

    formater_extended = LOGCONF['formatters']['extended']

    # If we put the name of logging it in LOGCONF, the file is created when
    # logging system is initialized, we don't want that
    # handler_file = LOGCONF['handlers']['processing_file']['filename']
    handler_file = "processing.log"
    logger_control = dict(
        default=__name__,
        root_levels=['numina'],
        logfile=handler_file,
        format=formater_extended['format'],
        enabled=True
    )
    request_params['logger_control'] = logger_control

    task = datastore.backend.new_task(request, request_params)

    # We should control here any possible failure
    try:
        return run_task_reduce(task, datastore)
    finally:
        datastore.store_task(task)


def config_recipe_logger(root_level_logger, ref_logger='numina'):
    """Configure root level logger for recipe.

    The recipe formatter is 'detailed'
    The logger is set to DEBUG (this is needed by the FileHandler later
    The StreamHandler is set to the same level as the logger of numina

    """
    numina_logger = logging.getLogger(ref_logger)
    recipe_logger = logging.getLogger(root_level_logger)
    recipe_logger.setLevel(logging.DEBUG)
    recipe_logger.propagate = False

    # create formatter
    formater_dd = LOGCONF['formatters']['detailed']

    detailed_formatter = logging.Formatter(fmt=formater_dd.get('format'))
    # create handler, ingnoring configuration here
    #handerl_dd = numina_cli_logconf['handlers']['detailed_console']
    #handerl_dd_level =
    sh = logging.StreamHandler()
    sh.setLevel(numina_logger.getEffectiveLevel())
    sh.setFormatter(detailed_formatter)
    recipe_logger.addHandler(sh)
    return recipe_logger


def run_task_reduce(task, datastore):

    obsid = task.request_params['oblock_id']
    configuration = task.request_params["instrument_configuration"]
    as_mode = task.request_params["mode"]

    _logger.info(f"procesing OB with id={obsid}")

    workenv = datastore.create_workenv(task)
    task.request_runinfo["results_dir"] = workenv.resultsdir_rel
    task.request_runinfo["work_dir"] = workenv.workdir_rel
    workenv.sane_work()

    # Roll back to cwd after leaving the context
    with working_directory(workenv.datadir):

        obsres = datastore.backend.obsres_from_oblock_id(
            obsid, as_mode=as_mode, configuration=configuration
        )
        # Merge requirements passed from above
        obsres.requirements.update(task.request_params['requirements'])
        obsres.pipeline = task.request_params["pipeline"]
        _logger.debug("pipeline is %s", obsres.pipeline)

        recipe = datastore.backend.search_recipe_from_ob(obsres)
        _logger.debug('recipe class is %s', recipe.__class__)

        recipe.intermediate_results = task.request_params["intermediate_results"]
        recipe.validate_inputs = task.request_params["validate_inputs"]
        recipe.validate_results = task.request_params["validate_results"]

        # Update runinfo
        recipe.runinfo['runner'] = task.request_runinfo['runner']
        recipe.runinfo['runner_version'] = task.request_runinfo['runner_version']
        recipe.runinfo['task_id'] = task.id
        recipe.runinfo['data_dir'] = workenv.datadir
        recipe.runinfo['work_dir'] = workenv.workdir
        recipe.runinfo['results_dir'] = workenv.resultsdir
        recipe.runinfo['intermediate_results'] = task.request_params["intermediate_results"]

        _logger.debug('recipe created')

        try:
            rinput = recipe.build_recipe_input(obsres, datastore.backend)
        except (ValueError, numina.exceptions.ValidationError) as err:
            _logger.error("During recipe input construction")
            _logger.error("%s", err)
            raise

        _logger.debug('recipe input created')
        # Show the actual inputs
        for key, req in recipe.requirements().items():
            v = getattr(rinput, key)
            _logger.debug("recipe requires %r, value is %s", key, v)

        for key, val in obsres.requirements.items():
            if key not in recipe.requirements():
                _logger.warning(f'"{key}: {val}" present in OB requirements, but not used')

        for req in recipe.products().values():
            _logger.debug('recipe provides %s, %s', req.type.__class__.__name__, req.description)

    # Load recipe control and recipe parameters from file
    task.request_runinfo['instrument'] = obsres.instrument
    task.request_runinfo['pipeline'] = obsres.pipeline
    task.request_runinfo['mode'] = obsres.mode
    task.request_runinfo['recipe_class'] = recipe.__class__.__name__
    task.request_runinfo['recipe_fqn'] = fully_qualified_name(recipe.__class__)
    task.request_runinfo['recipe_version'] = recipe.__version__
    root_level_logger = task.request_runinfo['recipe_fqn'].split('.')[0]

    # Copy files
    if task.request_params["copy_files"]:
        install_action = 'copy'
        _logger.debug('copy files to work directory')
    else:
        install_action = 'link'
        _logger.debug('link files to work directory')

    workenv.installfiles_stage1(obsres, action=install_action)
    workenv.installfiles_stage2(rinput, action=install_action)
    workenv.adapt_obsres(obsres)

    logger_control = task.request_params['logger_control']
    # configure recipe root level
    config_recipe_logger(root_level_logger)
    logger_control['root_levels'].append(root_level_logger)
    # Add file logging
    with logger_manager(logger_control, workenv.resultsdir):
        with working_directory(workenv.workdir):
            completed_task = run_recipe_timed(task, recipe, rinput)

    # datastore.store_task(completed_task)
    return completed_task


@contextlib.contextmanager
def logger_manager(logger_control, result_dir):
    """"Add a FileHandler to existing loggers

    We add a FileHandler to the loggers named in
    logger_control['root_levels'] if
    logger_control['root_levels'] is True
    with level DEBUG

    After exiting the context, the handlers are removed
    """
    # Creating custom logger file
    if logger_control['root_levels']:
        logfile = os.path.join(result_dir, logger_control['logfile'])
        logformat = logger_control['format']
        recipe_formatter = logging.Formatter(logformat)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(recipe_formatter)
    else:
        fh = logging.NullHandler()

    root_levels = logger_control['root_levels']
    recipe_loggers =  []
    for lname in root_levels:
        recipe_logger = logging.getLogger(lname)
        recipe_logger.setLevel(logging.DEBUG)
        recipe_logger.addHandler(fh)
        recipe_loggers.append(recipe_logger)
    try:
        yield recipe_loggers
    finally:
        for recipe_logger in recipe_loggers:
            recipe_logger.removeHandler(fh)


def run_recipe_timed(task, recipe, rinput):
    """Run the recipe and count the time it takes."""
    _logger.info('running recipe')
    task.state = 1
    task.time_start = datetime.datetime.now()
    #
    try:
        task.result = recipe(rinput)
        task.state = 2
    except Exception:
        task.state = 3
        raise
    finally:
        task.time_end = datetime.datetime.now()
    return task
