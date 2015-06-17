#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

'''User command line interface of Numina.'''

import sys
import os
import logging
import datetime

import yaml

from numina import __version__
from numina.core import obsres_from_dict
from numina.core import InstrumentConfiguration
from numina.core import import_object
from numina.core import fully_qualified_name
from numina.core.recipeinput import RecipeInputBuilder

from .helpers import ProcessingTask, WorkEnvironment, DiskStorageDefault

_logger = logging.getLogger("numina")

from numina.core.dal.dictdal import DictDAL, BaseDictDAL
from numina.core.recipeinput import RecipeInputBuilderCLI as RecipeInputBuilder

class Dict2DAL(BaseDictDAL):
    def __init__(self, obtable, base, basedir='_data'):

        prod_table = base['products']
        req_table= base['requirements']
        super(Dict2DAL, self).__init__(obtable, prod_table, req_table)
        self.basedir = basedir

def process_format_version_0(loaded_data):
    return 0

def process_format_version_1(loaded_obs, loaded_data):
    return Dict2DAL(loaded_obs, loaded_data)

#----------------------------------------

def load_from_obsres(obsres, args):
    _logger.info("Identifier of the observation result: %d", obsres.id)
    ins_name = obsres.instrument
    _logger.info("instrument name: %s", ins_name)
    my_ins = args.drps.get(ins_name)
    if my_ins is None:
        _logger.error('instrument %r does not exist', ins_name)
        sys.exit(1)

    _logger.debug('instrument is %s', my_ins)
    # Load configuration from the command line
    if args.insconf is not None:
        _logger.debug("configuration from CLI is %r", args.insconf)
        ins_conf = args.insconf
    else:
        ins_conf = obsres.configuration

    _logger.info('loading instrument configuration %r', ins_conf)
    my_ins_conf = my_ins.configurations.get(ins_conf)

    if my_ins_conf:
        _logger.debug('instrument configuration object is %r', my_ins_conf)
    else:
        # Trying to open a file
        try:
            with open(ins_conf) as fd:
                values = yaml.load(fd)
            if values is None:
                _logger.warning('%r is empty', ins_conf)
                values = {}
            else:
                # FIXME this file should be validated
                _logger.warning('loading unvalidated instrument configuration')
                _logger.warning('you were warned')

            ins_conf = values.get('name', ins_conf)
            my_ins_conf = InstrumentConfiguration(ins_conf, values)

            # The new configuration must not overwrite existing configurations
            if ins_conf not in my_ins.configurations:
                my_ins.configurations[ins_conf] = my_ins_conf
            else:
                _logger.error('a configuration already exists %r, exiting',
                              ins_conf)
            sys.exit(1)

        except IOError:
            _logger.error('instrument configuration %r does not exist',
                          ins_conf)
            sys.exit(1)

    # Loading the pipeline
    if args.pipe_name is not None:
        _logger.debug("pipeline from CLI is %r", args.pipe_name)
        pipe_name = args.pipe_name
    else:
        pipe_name = obsres.pipeline
        _logger.debug("pipeline from ObsResult is %r", pipe_name)

    my_pipe = my_ins.pipelines.get(pipe_name)
    if my_pipe is None:
        _logger.error(
            'instrument %r does not have pipeline named %r',
            ins_name,
            pipe_name
            )
        sys.exit(1)

    _logger.info('loading pipeline %r', pipe_name)
    _logger.debug('pipeline object is %s', my_pipe)

    obs_mode = obsres.mode
    _logger.info("observing mode: %r", obs_mode)

    recipe_fqn = my_pipe.recipes.get(obs_mode)
    if recipe_fqn is None:
        _logger.error(
            'pipeline %r does not have recipe to process %r obs mode',
            pipe_name, obs_mode
            )
        sys.exit(1)
    return recipe_fqn, pipe_name, my_ins_conf, ins_conf


def mode_run_common(args, mode):
    '''Observing mode processing mode of numina.'''

    # Directories with relevant data
    workenv = WorkEnvironment(
        args.basedir,
        workdir=args.workdir,
        resultsdir=args.resultsdir,
        datadir=args.datadir
        )

    # Loading observation result if exists
    loaded_obs = {}
    _logger.info("Loading observation results from %r", args.obsresult)
    with open(args.obsresult) as fd:
        for doc in yaml.load_all(fd):
            loaded_obs[doc['id']] = doc

    _logger.info('reading control from %s', args.reqs)
    with open(args.reqs, 'r') as fd:
        loaded_data = yaml.load(fd)

    control_format = loaded_data.get('version', 0)

    if control_format == 0:
        dal = process_format_version_0(loaded_data)
    elif control_format == 1:
        dal = process_format_version_1(loaded_obs, loaded_data)
    else:
        print('Unsupported format', control_format, 'in', arg.reqs)
        sys.exit(1)

    # Start processing

    cwd = os.getcwd()
    os.chdir(workenv.datadir)
    
    pipe_name = 'default'
    obsres = dal.obsres_from_oblock_id(1)
    
    recipeclass = dal.search_recipe_from_ob(obsres, pipe_name)
    _logger.debug('recipe class is %s', recipeclass)

    _logger.debug('recipe input builder class is %s', RecipeInputBuilder)
    ri_builder = RecipeInputBuilder(recipeclass, dal)
    _logger.debug('create RecipeInputBuilder %s', ri_builder)

    _logger.debug('build recipe input object')
    rinput = ri_builder.buildRI(obsres)

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
    for key, val in recipeclass.Requirements.iteritems():
        _logger.info("recipe requires %r", val.type.__class__)
        _logger.info("%r is %r", val.dest, getattr(rinput, val.dest))

    _logger.debug('parsing products')
    for req in recipeclass.Result.values():
        _logger.info('recipe provides %r', req)

    task = ProcessingTask(obsres=obsres, insconf={})
    task.runinfo['pipeline'] = pipe_name
    task.runinfo['recipe'] = recipeclass.__name__
    task.runinfo['recipe_full_name'] = fully_qualified_name(recipeclass)
    task.runinfo['runner'] = 'numina'
    task.runinfo['runner_version'] = __version__
    task.runinfo['data_dir'] = workenv.datadir
    task.runinfo['work_dir'] = workenv.workdir
    task.runinfo['results_dir'] = workenv.resultsdir
    task.runinfo['recipe_version'] = recipe.__version__

    completed_task = run_recipe(
        recipe=recipe,
        task=task, rinput=rinput,
        workenv=workenv, task_control=task_control
        )

    where = DiskStorageDefault(resultsdir=workenv.resultsdir)

    where.store(completed_task)


def create_recipe_file_logger(logger, logfile, logformat):
    '''Redirect Recipe log messages to a file.'''
    recipe_formatter = logging.Formatter(logformat)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(recipe_formatter)
    return fh


def run_recipe(recipe, task, rinput, workenv, task_control):
    '''Recipe execution mode of numina.'''

    # Creating custom logger file
    DEFAULT_RECIPE_LOGGER = 'numina.recipes'
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

    try:
        csd = os.getcwd()
        _logger.debug('cwd to workdir')
        os.chdir(workenv.workdir)
        completed_task = run_recipe_timed(recipe, rinput, task)

        return completed_task

    finally:
        _logger.debug('cwd to original path: %r', csd)
        os.chdir(csd)
        recipe_logger.removeHandler(fh)


def run_recipe_timed(recipe, rinput, task):
    '''Run the recipe and count the time it takes.'''
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

