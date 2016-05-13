#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""User command line interface of Numina."""

import sys
import os
import logging
import datetime

import yaml

from numina.dal.dictdal import BaseDictDAL

from .helpers import ProcessingTask, WorkEnvironment, DiskStorageDefault
from .clidal import process_format_version_0

DEFAULT_RECIPE_LOGGER = 'numina.recipes'

_logger = logging.getLogger("numina")


class Dict2DAL(BaseDictDAL):
    def __init__(self, obtable, base):

        prod_table = base['products']

        if 'parameters' in base:
            req_table = base['parameters']
        else:
            req_table = base['requirements']

        super(Dict2DAL, self).__init__(obtable, prod_table, req_table)


def process_format_version_1(loaded_obs, loaded_data):
    return Dict2DAL(loaded_obs, loaded_data)


def mode_run_common(args, mode):
    # FIXME: implement 'recipe' run mode
    if mode == 'rec':
        print('Mode not implemented yet')
        return 1
    elif mode == 'obs':
        return mode_run_common_obs(args)
    else:
        raise ValueError('Not valid run mode {0}'.format(mode))


def mode_run_common_obs(args):
    """Observing mode processing mode of numina."""

    # Directories with relevant data
    workenv = WorkEnvironment(args.basedir,
                              workdir=args.workdir,
                              resultsdir=args.resultsdir,
                              datadir=args.datadir
                              )

    # Loading observation result if exists
    loaded_obs = {}
    _logger.info("Loading observation results from %r", args.obsresult)
    loaded_ids = []
    with open(args.obsresult) as fd:
        for doc in yaml.load_all(fd):
            loaded_ids.append(doc['id'])
            loaded_obs[doc['id']] = doc

    _logger.info('reading control from %s', args.reqs)
    with open(args.reqs, 'r') as fd:
        loaded_data = yaml.load(fd)

    control_format = loaded_data.get('version', 0)

    if control_format == 0:
        dal = process_format_version_0(loaded_obs, loaded_data)
    elif control_format == 1:
        dal = process_format_version_1(loaded_obs, loaded_data)
    else:
        print('Unsupported format', control_format, 'in', args.reqs)
        sys.exit(1)

    _logger.info('control format version %d', control_format)
    # Start processing

    cwd = os.getcwd()
    os.chdir(workenv.datadir)

    # Only the first, for the moment
    if args.insconf:
        _logger.debug("instrument configuration from CLI is %r", args.insconf)

    obsres = dal.obsres_from_oblock_id(loaded_ids[0], configuration=args.insconf)

    _logger.debug("pipeline from CLI is %r", args.pipe_name)
    pipe_name = args.pipe_name
    recipeclass = dal.search_recipe_from_ob(obsres, pipe_name)
    _logger.debug('recipe class is %s', recipeclass)

    rinput = recipeclass.build_recipe_input(obsres, dal, pipeline=pipe_name)
    _logger.debug('recipe input created')

    # Show the actual inputs
    _logger.debug('parsing requirements')
    for key in recipeclass.requirements():
        v = getattr(rinput, key)
        _logger.info("recipe requires %r value is %s", key, v)

    _logger.debug('parsing products')
    for req in recipeclass.products().values():
        _logger.info('recipe provides %s, %s', req.type, req.description)

    os.chdir(cwd)

    recipe = recipeclass()
    _logger.debug('recipe created')

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
    _logger.debug('parsing requirements')
    for key in recipeclass.requirements().values():
        _logger.info("recipe requires %r", key)

    _logger.debug('parsing products')
    for req in recipeclass.products().values():
        _logger.info('recipe provides %r', req)

    runinfo = {
        'taskid': loaded_ids[0],
        'pipeline': pipe_name,
        'recipeclass': recipeclass,
        'workenv': workenv,
        'recipe_version': recipe.__version__,
        'instrument_configuration': args.insconf
    }

    task = ProcessingTask(obsres, runinfo)

    # Copy files
    _logger.debug('copy files to work directory')
    workenv.sane_work()
    workenv.copyfiles_stage1(obsres)
    workenv.copyfiles_stage2(rinput)

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

    csd = os.getcwd()
    try:
        _logger.debug('cwd to workdir')
        os.chdir(workenv.workdir)
        completed_task = run_recipe_timed(recipe, rinput, task)

        return completed_task

    finally:
        _logger.debug('cwd to original path: %r', csd)
        os.chdir(csd)
        recipe_logger.removeHandler(fh)


def run_recipe_timed(recipe, rinput, task):
    """Run the recipe and count the time it takes."""
    TIMEFMT = '%FT%T'
    _logger.info('running recipe')
    now1 = datetime.datetime.now()
    task.runinfo['time_start'] = now1.strftime(TIMEFMT)
    #

    result = recipe.run(rinput)
    _logger.info('result: %r', result)
    task.result = result
    #
    now2 = datetime.datetime.now()
    task.runinfo['time_end'] = now2.strftime(TIMEFMT)
    task.runinfo['time_running'] = now2 - now1
    return task
