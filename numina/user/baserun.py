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
import errno
import shutil
import pickle

import six
import yaml

from numina import __version__
import numina.exceptions
from numina.util.context import working_directory
from numina.util.fqn import fully_qualified_name
from numina.util.jsonencoder import ExtEncoder
from numina.types.frame import DataFrameType
from numina.util.context import working_directory


_logger = logging.getLogger(__name__)


class ReductionBlock(object):
    def __init__(self):
        self.id = 1
        self.instrument = None
        self.mode = None
        self.pipeline = 'default'
        self.instrument_configuration = None
        self.requirements = {}


def run_job(datastore, obsid, copy_files=True):
    """Observing mode processing mode of numina."""

    request = 'reduce'
    request_params = {}

    request_params['oblock_id'] = obsid
    request_params["pipeline"] = 'default' #  args.pipe_name
    request_params["instrument_configuration"] = 'default'  # args.insconf
    request_params["intermediate_results"] = True
    request_params["copy_files"] = copy_files

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
        return run_task_reduce(datastore, task)
    finally:
        datastore.store_task(task)


def run_task_reduce(datastore, task):

    task.request_runinfo['runner'] = 'numina'
    task.request_runinfo['runner_version'] = __version__

    obsid = task.request_params['oblock_id']
    configuration = task.request_params["instrument_configuration"]

    _logger.info("procesing OB with id={}".format(obsid))

    workenv = datastore.create_workenv(task)
    task.request_runinfo["results_dir"] = workenv.resultsdir_rel
    task.request_runinfo["work_dir"] = workenv.workdir_rel

    # Roll back to cwd after leaving the context
    with working_directory(workenv.datadir):

        obsres = datastore.backend.obsres_from_oblock_id(
            obsid, configuration=configuration
        )

        pipe_name = task.request_params["pipeline"]
        _logger.debug("pipeline is %s", pipe_name)
        obsres.pipeline = pipe_name
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
            # sys.exit(0)

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
    workenv.sane_work()
    if task.request_params["copy_files"]:
        _logger.debug('copy files to work directory')
        workenv.copyfiles_stage1(obsres)
        workenv.copyfiles_stage2(rinput)
        workenv.adapt_obsres(obsres)

    logger_control = task.request_params['logger_control']
    with logger_manager(logger_control, workenv.resultsdir):
        with working_directory(workenv.workdir):
            completed_task = run_recipe_timed(task, recipe, rinput)

    datastore.store_task(completed_task)
    return completed_task


def run_reduce(datastore, obsid, copy_files=True):
    """Observing mode processing mode of numina."""

    request = 'reduce'
    request_params = {}

    rb = ReductionBlock()
    rb.id = obsid

    request_params['oblock_id'] = rb.id
    request_params["pipeline"] = rb.pipeline
    request_params["instrument_configuration"] = rb.instrument_configuration

    logger_control = dict(
        default=__name__,
        logfile='processing.log',
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        enabled=True
    )
    request_params['logger_control'] = logger_control

    task = datastore.backend.new_task(request, request_params)
    task.request = request
    task.request_params = request_params

    task.request_runinfo['runner'] = 'numina'
    task.request_runinfo['runner_version'] = __version__

    _logger.info("procesing OB with id={}".format(obsid))
    workenv = datastore.create_workenv(task)

    task.request_runinfo["results_dir"] = workenv.resultsdir_rel
    task.request_runinfo["work_dir"] = workenv.workdir_rel

    # Roll back to cwd after leaving the context
    with working_directory(workenv.datadir):

        obsres = datastore.backend.obsres_from_oblock_id(obsid, configuration=rb.instrument_configuration)

        _logger.debug("pipeline from CLI is %r", rb.pipeline)
        pipe_name = rb.pipeline
        obsres.pipeline = pipe_name
        recipe = datastore.backend.search_recipe_from_ob(obsres)
        _logger.debug('recipe class is %s', recipe.__class__)

        # Enable intermediate results by default
        _logger.debug('enable intermediate results')
        recipe.intermediate_results = True

        # Update runinfo
        _logger.debug('update recipe runinfo')
        recipe.runinfo['runner'] = 'numina'
        recipe.runinfo['runner_version'] = __version__
        recipe.runinfo['task_id'] = task.id
        recipe.runinfo['data_dir'] = workenv.datadir
        recipe.runinfo['work_dir'] = workenv.workdir
        recipe.runinfo['results_dir'] = workenv.resultsdir

        _logger.debug('recipe created')

        try:
            rinput = recipe.build_recipe_input(obsres, datastore.backend)
        except (ValueError, numina.exceptions.ValidationError) as err:
            _logger.error("During recipe input construction")
            _logger.error("%s", err)
            raise
            # sys.exit(0)

        _logger.debug('recipe input created')
        # Show the actual inputs
        for key in recipe.requirements():
            v = getattr(rinput, key)
            _logger.debug("recipe requires %r, value is %s", key, v)

        for req in recipe.products().values():
            _logger.debug('recipe provides %s, %s', req.type.__class__.__name__, req.description)

    # Load recipe control and recipe parameters from file
    task.request_runinfo['instrument'] = obsres.instrument
    task.request_runinfo['mode'] = obsres.mode
    task.request_runinfo['recipe_class'] = recipe.__class__.__name__
    task.request_runinfo['recipe_fqn'] = fully_qualified_name(recipe.__class__)
    task.request_runinfo['recipe_version'] = recipe.__version__

    # Copy files
    workenv.sane_work()
    if copy_files:
        _logger.debug('copy files to work directory')
        workenv.copyfiles_stage1(obsres)
        workenv.copyfiles_stage2(rinput)
        workenv.adapt_obsres(obsres)

    with logger_manager(logger_control, workenv.resultsdir):
        with working_directory(workenv.workdir):
            completed_task = run_recipe_timed(task, recipe, rinput)

    datastore.store_task(completed_task)
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
    now1 = datetime.datetime.now()
    task.state = 1
    task.time_start = now1
    #
    result = recipe(rinput)
    _logger.info('result: %r', result)
    task.result = result
    #
    now2 = datetime.datetime.now()
    task.state = 2
    task.time_end = now2
    return task
