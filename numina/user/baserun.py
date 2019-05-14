#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

from __future__ import print_function

import datetime
import logging
import os
import contextlib


import numina.exceptions
from numina.util.fqn import fully_qualified_name
from numina.util.context import working_directory


_logger = logging.getLogger(__name__)


def run_reduce(datastore, obsid, as_mode=None, requirements=None, copy_files=False):
    """Observing mode processing mode of numina."""

    request = 'reduce'
    request_params = dict()

    request_params['oblock_id'] = obsid
    request_params["pipeline"] = 'default' #  args.pipe_name
    request_params["instrument_configuration"] = 'default'  # args.insconf
    request_params["intermediate_results"] = True
    request_params["copy_files"] = copy_files
    requirements = {} if requirements is None else requirements
    request_params["requirements"] = requirements

    logger_control = dict(
        default=__name__,
        logfile='processing.log',
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        enabled=True
    )
    request_params['logger_control'] = logger_control

    task = datastore.backend.new_task(request, request_params)

    # We should control here any possible failure
    try:
        return run_task_reduce(task, datastore)
    finally:
        datastore.store_task(task)


def run_task_reduce(task, datastore):

    obsid = task.request_params['oblock_id']
    configuration = task.request_params["instrument_configuration"]

    _logger.info("procesing OB with id={}".format(obsid))

    workenv = datastore.create_workenv(task)
    task.request_runinfo["results_dir"] = workenv.resultsdir_rel
    task.request_runinfo["work_dir"] = workenv.workdir_rel
    workenv.sane_work()

    # Roll back to cwd after leaving the context
    with working_directory(workenv.datadir):

        obsres = datastore.backend.obsres_from_oblock_id(
            obsid, configuration=configuration
        )
        # Merge requirements passed from above
        obsres.requirements.update(task.request_params['requirements'])
        obsres.pipeline = task.request_params["pipeline"]
        _logger.debug("pipeline is %s", obsres.pipeline)

        recipe = datastore.backend.search_recipe_from_ob(obsres)
        _logger.debug('recipe class is %s', recipe.__class__)

        recipe.intermediate_results = task.request_params["intermediate_results"]

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
        for key in recipe.requirements():
            v = getattr(rinput, key)
            _logger.debug("recipe requires %r, value is %s", key, v)

        for req in recipe.products().values():
            _logger.debug('recipe provides %s, %s', req.type.__class__.__name__, req.description)

    # Load recipe control and recipe parameters from file
    task.request_runinfo['instrument'] = obsres.instrument
    task.request_runinfo['pipeline'] = obsres.pipeline
    task.request_runinfo['mode'] = obsres.mode
    task.request_runinfo['recipe_class'] = recipe.__class__.__name__
    task.request_runinfo['recipe_fqn'] = fully_qualified_name(recipe.__class__)
    task.request_runinfo['recipe_version'] = recipe.__version__

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
    with logger_manager(logger_control, workenv.resultsdir):
        with working_directory(workenv.workdir):
            completed_task = run_recipe_timed(task, recipe, rinput)

    # datastore.store_task(completed_task)
    return completed_task


def create_recipe_file_logger(logger, logfile, logformat):
    """Redirect Recipe log messages to a file."""
    recipe_formatter = logging.Formatter(logformat)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(recipe_formatter)
    return fh


@contextlib.contextmanager
def logger_manager(logger_control, result_dir):

    # Creating custom logger file
    recipe_logger = logging.getLogger(logger_control['default'])
    if logger_control['enabled']:
        logfile = os.path.join(result_dir, logger_control['logfile'])
        logformat = logger_control['format']
        fh = create_recipe_file_logger(recipe_logger, logfile, logformat)
    else:
        fh = logging.NullHandler()

    recipe_logger.addHandler(fh)

    try:
        yield recipe_logger
    finally:
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
        _logger.info('result: %r', task.result)
    except Exception:
        task.state = 3
        raise
    finally:
        task.time_end = datetime.datetime.now()
    return task



