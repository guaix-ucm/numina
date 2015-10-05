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


def load_control(rfile):
    # Load recipe control and recipe parameters from file
    task_control = dict(requirements={}, products={}, logger=logger_control)
    task_control_loaded = {}

    # Read task_control from args.reqs
    _logger.info('reading task control from %s', rfile)
    with open(rfile, 'r') as fd:
        task_control_loaded = yaml.load(fd)

    # Here, check dialect of task_control_loaded

    # Populate task_control
    for key in task_control:
        if key in task_control_loaded:
            task_control[key].update(task_control_loaded[key])

    return task_control


def load_from_obsres(obsres, drps, args):
    _logger.info("Identifier of the observation result: %d", obsres.id)
    ins_name = obsres.instrument
    _logger.info("instrument name: %s", ins_name)
    my_ins = drps.get(ins_name)
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
    obsres = None
    if args.obsresult is not None:
        _logger.info("Loading observation result from %r", args.obsresult)

        with open(args.obsresult) as fd:
            obsres = obsres_from_dict(yaml.load(fd))

        _logger.debug('images in observation result')
        for v in obsres.images:
            _logger.debug('%r', v)

    if mode == 'obs':
        drps = init_drp_system()
        values = load_from_obsres(obsres, drps, args)
        recipe_fqn, pipe_name, my_ins_conf, ins_conf = values
    elif mode == 'rec':
        my_ins_conf = None
        pipe_name = None
        ins_conf = None
        recipe_fqn = args.recipe
    else:
        raise ValueError('mode must be one of "obs\rec"')

    _logger.debug('recipe fqn is %s', recipe_fqn)
    recipeclass = import_object(recipe_fqn)
    _logger.debug('recipe class is %s', recipeclass)
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
    task_control_loaded = {}

    # Read task_control from args.reqs
    if args.reqs is not None:
        _logger.info('reading task control from %s', args.reqs)
        with open(args.reqs, 'r') as fd:
            task_control_loaded = yaml.load(fd)
            # backward compatibility with 'parameters' key
            if 'parameters' in task_control_loaded:
                if 'requirements' not in task_control_loaded:
                    _logger.warning("'requirements' missing, "
                                    "using 'parameters'")
                    task_control_loaded['requirements'] = \
                        task_control_loaded['parameters']

    # Populate task_control
    for key in task_control:
        if key in task_control_loaded:
            task_control[key].update(task_control_loaded[key])

    task_control['requirements']['obresult'] = obsres
    task_control['requirements']['insconf'] = my_ins_conf

    # Build the recipe input data structure
    # and copy needed files to workdir
    _logger.debug('recipe input builder class is %s', RecipeInputBuilder)
    rib = RecipeInputBuilder()
    try:
        rinput = rib.build(workenv, recipeclass, task_control['requirements'])
    except ValueError as error:
        _logger.error('%s, exiting', error)
        sys.exit(1)
    except (OSError, IOError) as exception:
        _logger.error('%s, exiting', exception)
        sys.exit(1)

    _logger.debug('parsing requirements')
    for key, val in recipeclass.requirements().iteritems():
        _logger.info("recipe requires %r", val.type.__class__)
        _logger.info("%r is %r", val.dest, getattr(rinput, val.dest))

    _logger.debug('parsing products')
    for req in recipeclass.products().values():
        _logger.info('recipe provides %r', req)

    task = ProcessingTask(obsres=obsres, insconf=ins_conf)
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

