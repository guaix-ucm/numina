#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import sys
import os
import logging
import datetime

import yaml

from numina import __version__
import numina.drps
import numina.exceptions
from numina.dal.dictdal import HybridDAL
from numina.util.context import working_directory

from .helpers import ProcessingTask, WorkEnvironment, DiskStorageDefault


DEFAULT_RECIPE_LOGGER = 'numina.recipes'

_logger = logging.getLogger("numina")


def process_format_version_1(loaded_obs, loaded_data, loaded_data_extra=None):
    drps = numina.drps.get_system_drps()
    return HybridDAL(drps, loaded_obs, loaded_data, loaded_data_extra)


def mode_run_common(args, extra_args, mode):
    # FIXME: implement 'recipe' run mode
    if mode == 'rec':
        print('Mode not implemented yet')
        return 1
    elif mode == 'obs':
        return mode_run_common_obs(args, extra_args)
    else:
        raise ValueError('Not valid run mode {0}'.format(mode))


def mode_run_common_obs(args, extra_args):
    """Observing mode processing mode of numina."""

    # Loading observation result if exists
    loaded_obs = []
    for obfile in args.obsresult:
        _logger.info("Loading observation results from %r", obfile)

        with open(obfile) as fd:
            for doc in yaml.load_all(fd):
                enabled = doc.get('enabled', True)
                docid = doc['id']
                if enabled:
                    _logger.debug("load observation result with id %s", docid)
                else:
                    _logger.debug("skip observation result with id %s", docid)
                
                loaded_obs.append(doc)
    if args.reqs:
        _logger.info('reading control from %s', args.reqs)
        with open(args.reqs, 'r') as fd:
            loaded_data = yaml.load(fd)
    else:
        _logger.info('no control file')
        loaded_data = {}

    if extra_args.extra_control:
        _logger.info('extra control %s', extra_args.extra_control)
        loaded_data_extra = parse_as_yaml(extra_args.extra_control)
    else:
        loaded_data_extra = None

    control_format = loaded_data.get('version', 0)
    _logger.info('control format version %d', control_format)

    # FIXME: DAL and WorkEnvironment
    # DAL and WorkEnvironment
    # must share its information
    #
    if control_format == 1:
        dal = process_format_version_1(loaded_obs, loaded_data, loaded_data_extra)
    else:
        print('Unsupported format', control_format, 'in', args.reqs)
        sys.exit(1)

    # Start processing
    for obid in dal.search_session_ids():
        # Directories with relevant data
        _logger.info("procesing OB with id={}".format(obid))
        workenv = WorkEnvironment(obid,
                                  args.basedir,
                                  workdir=args.workdir,
                                  resultsdir=args.resultsdir,
                                  datadir=args.datadir
                                  )

        # Roll back to cwd after leaving the context
        with working_directory(workenv.datadir):

            obsres = dal.obsres_from_oblock_id(obid, configuration=args.insconf)

            _logger.debug("pipeline from CLI is %r", args.pipe_name)
            pipe_name = args.pipe_name
            obsres.pipeline = pipe_name
            recipe = dal.search_recipe_from_ob(obsres)
            _logger.debug('recipe class is %s', recipe.__class__)

            # Enable intermediate results by default
            _logger.debug('enable intermediate results')
            recipe.intermediate_results = True

            # Update runinfo
            _logger.debug('update recipe runinfo')
            recipe.runinfo['runner'] = 'numina'
            recipe.runinfo['runner_version'] = '1'
            recipe.runinfo['taskid'] = obid
            recipe.runinfo['data_dir'] = workenv.datadir
            recipe.runinfo['work_dir'] = workenv.workdir
            recipe.runinfo['results_dir'] = workenv.resultsdir

            _logger.debug('recipe created')

            try:
                rinput = recipe.build_recipe_input(obsres, dal)
            except (ValueError, numina.exceptions.ValidationError) as err:
                _logger.error("During recipe input construction")
                _logger.error("%s", err)
                sys.exit(0)
            _logger.debug('recipe input created')

            # Show the actual inputs
            for key in recipe.requirements():
                v = getattr(rinput, key)
                _logger.debug("recipe requires %r, value is %s", key, v)

            for req in recipe.products().values():
                _logger.debug('recipe provides %s, %s', req.type.__class__.__name__, req.description)

        # Logging and task control
        logger_control = dict(
            logfile='processing.log',
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            enabled=True
            )

        # Load recipe control and recipe parameters from file
        task_control = dict(requirements={}, products={}, logger=logger_control)

        # Build the recipe input data structure
        # and copy needed files to workdir
        runinfo = {
            'taskid': obid,
            'obsid': obid,
            'pipeline': pipe_name,
            'recipeclass': recipe.__class__,
            'workenv': workenv,
            'recipe_version': recipe.__version__,
            'runner': 'numina',
            'runner_version': __version__,
            'instrument_configuration': args.insconf
        }

        task = ProcessingTask(obsres, runinfo)

        # Copy files
        if args.copy_files:
            _logger.debug('copy files to work directory')
            workenv.sane_work()
            workenv.copyfiles_stage1(obsres)
            workenv.copyfiles_stage2(rinput)
            workenv.adapt_obsres(obsres)

        completed_task = run_recipe(recipe=recipe, task=task, rinput=rinput,
                                    workenv=workenv, task_control=task_control)

        where = DiskStorageDefault(resultsdir=workenv.resultsdir)
        where.store(completed_task)


def create_recipe_file_logger(logger, logfile, logformat):
    """Redirect Recipe log messages to a file."""
    recipe_formatter = logging.Formatter(logformat)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(recipe_formatter)
    return fh


def run_recipe(recipe, task, rinput, workenv, task_control):
    """Recipe execution mode of numina."""

    # Creating custom logger file
    recipe_logger = logging.getLogger(DEFAULT_RECIPE_LOGGER)

    logger_control = task_control['logger']
    if logger_control['enabled']:
        logfile = os.path.join(workenv.resultsdir, logger_control['logfile'])
        logformat = logger_control['format']
        _logger.debug('creating file logger %r from Recipe logger', logfile)
        fh = create_recipe_file_logger(recipe_logger, logfile, logformat)
    else:
        fh = logging.NullHandler()

    recipe_logger.addHandler(fh)

    with working_directory(workenv.workdir):
        try:
            completed_task = run_recipe_timed(recipe, rinput, task)
            return completed_task
        finally:
            recipe_logger.removeHandler(fh)


def run_recipe_timed(recipe, rinput, task):
    """Run the recipe and count the time it takes."""
    TIMEFMT = '%FT%T'
    _logger.info('running recipe')
    now1 = datetime.datetime.now()
    task.runinfo['time_start'] = now1.strftime(TIMEFMT)
    #

    result = recipe(rinput)
    _logger.info('result: %r', result)
    task.result = result
    #
    now2 = datetime.datetime.now()
    task.runinfo['time_end'] = now2.strftime(TIMEFMT)
    task.runinfo['time_running'] = now2 - now1
    return task


def parse_as_yaml(strdict):
    """Parse a dictionary of strings as if yaml reads it"""
    interm = ""
    for key, val in strdict.items():
        interm = "%s: %s, %s" % (key, val, interm)
    fin = '{%s}' % interm

    return yaml.load(fin)