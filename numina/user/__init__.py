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

from __future__ import print_function

import logging.config
import sys
import argparse
import os
import errno
import shutil
import datetime
import ConfigParser as configparser
import inspect

import yaml

import numina.pipelines as namespace
from numina import __version__
from numina.core import RequirementParser, obsres_from_dict
from numina.core import FrameDataProduct, DataProduct
from numina.core import InstrumentConfiguration
from numina.core import init_drp_system, import_object
from numina.core.requirements import RequirementError
from numina.core.recipeinput import RecipeInputBuilder
from numina.core.products import ValidationError
from numina.xdgdirs import xdg_config_home
from .store import store

_logger = logging.getLogger("numina")

_logconf = {'version': 1,
  'formatters': {'simple': {'format': '%(levelname)s: %(message)s'},
                 'state': {'format': '%(asctime)s - %(message)s'},
                 'unadorned': {'format': '%(message)s'},
                 'detailed': {'format': '%(name)s %(levelname)s %(message)s'},
                 },
  'handlers': {'unadorned_console':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'unadorned',
                'level': 'DEBUG'
                 },
               'simple_console':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'simple',
                'level': 'DEBUG'
                 },
               'simple_console_warnings_only':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'simple',
                'level': 'WARNING'
                 },
               'detailed_console':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'detailed',
                'level': 'DEBUG'
                 },
               },
  'loggers': {'numina': {'handlers': ['simple_console'], 'level': 'DEBUG', 'propagate': False},
              'numina.recipes': {'handlers': ['detailed_console'], 'level': 'DEBUG', 'propagate': False},
              },
  'root': {'handlers': ['detailed_console'], 'level': 'NOTSET'}
}

class ProcessingTask(object):
    def __init__(self, obsres=None, insconf=None):
    
        self.observation = {}
        self.runinfo = {}
        
        if obsres:
            self.observation['mode'] = obsres.mode
            self.observation['observing_result'] = obsres.id
            self.observation['instrument'] = obsres.instrument
        else:
            self.observation['mode'] = None
            self.observation['observing_result'] = None
            self.observation['instrument'] = None
            
        if insconf:
            self.observation['instrument_configuration'] = insconf

class WorkEnvironment(object):
    def __init__(self, basedir, workdir=None, 
                 resultsdir=None, datadir=None):

        self.basedir = basedir
        
        if workdir is None:
            workdir = os.path.join(basedir, '_work')

        self.workdir = os.path.abspath(workdir)

        if resultsdir is None:
            resultsdir = os.path.join(basedir, '_results')

        self.resultsdir = os.path.abspath(resultsdir)

        if datadir is None:
            datadir = os.path.join(basedir, '_data')

        self.datadir = os.path.abspath(datadir)

    def sane_work(self):        
        make_sure_path_doesnot_exist(self.workdir)
        _logger.debug('check workdir for working: %r', self.workdir) 
        make_sure_path_exists(self.workdir)

        make_sure_path_doesnot_exist(self.resultsdir)
        _logger.debug('check resultsdir to store results %r', self.resultsdir)
        make_sure_path_exists(self.resultsdir)

    def copyfiles(self, obsres, reqs):

        _logger.info('copying files from %r to %r', self.datadir, self.workdir)

        if obsres:
            self.copyfiles_stage1(obsres)
                
        self.copyfiles_stage2(reqs)

    def copyfiles_stage1(self, obsres):
        _logger.debug('copying files from observation result')
        for f in obsres.frames:
            _logger.debug('copying %r to %r', f.filename, self.workdir)
            complete = os.path.abspath(os.path.join(self.datadir, f.filename))
            shutil.copy(complete, self.workdir)
            
    def copyfiles_stage2(self, reqs):
        _logger.debug('copying files from requirements')
        for _, req in reqs.__class__.__stored__.items():
            if isinstance(req.type, FrameDataProduct):
                value = getattr(reqs, req.dest)
                if value is not None:
                    _logger.debug('copying %r to %r', value.filename, self.workdir)
                    complete = os.path.abspath(os.path.join(self.datadir, value.filename))
                    shutil.copy(complete, self.workdir)
                    


def fully_qualified_name(obj, sep='.'):
    if inspect.isclass(obj):
        return obj.__module__ + sep + obj.__name__
    else:
        return obj.__module__ + sep + obj.__class__.__name__

def make_sure_path_doesnot_exist(path):
    try:
        shutil.rmtree(path)
    except (OSError, IOError) as exception:
        if exception.errno != errno.ENOENT:
            raise

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except (OSError, IOError) as exception:
        if exception.errno != errno.EEXIST:
            raise

def create_recipe_file_logger(logger, logfile, logformat):
    _logger.debug('creating file logger %r from Recipe logger', logfile)
    _recipe_formatter = logging.Formatter(logformat)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_recipe_formatter)
    return fh

def show_recipes(args):
    this_recipe_print = print_recipe
    if args.template:
        this_recipe_print = print_recipe_template

    for theins in args.drps.values():
        # Per instrument
        if not args.instrument or (args.instrument == theins.name):
            for pipe in theins.pipelines.values():
                for mode, recipe_fqn in pipe.recipes.items():
                    if not args.name or (recipe_fqn in args.name):
                        Cls = import_object(recipe_fqn)
                        this_recipe_print(Cls, name=recipe_fqn, insname=theins.name, pipename=pipe.name, modename=mode)

def show_observingmodes(args):
    for theins in args.drps.values():
        if not args.instrument or (args.instrument == theins.name):
            for mode in theins.modes:
                if not args.name or (mode.key in args.name):
                    print_obsmode(mode, theins)

def show_instruments(args):
    for theins in args.drps.values():
        if not args.name or (theins.name in args.name):
            print_instrument(theins, modes=args.om)

def print_recipe_template(recipe, name=None, insname=None, pipename=None, modename=None):

    def print_io(req):
        dispname = req.dest
        if getattr(req, 'default', None) is not None:
            return (dispname, req.default)
        elif isinstance(req.type, FrameDataProduct):
            return (dispname, dispname + '.fits')
        elif isinstance(req.type, DataProduct):
            return (dispname, getattr(req.type, 'default', None))
        else:
            return (dispname, None)

    # Create a dictionary with tamplates
    requires = {}
    optional = {}
    for req in recipe.__requires__:
        if req.dest is None:
            # FIXME: add warning or something here
            continue
        if req.hidden:
            # I Do not want to print it
            continue
        if req.optional:
            out = optional
        else:
            out = requires
    
        k, v = print_io(req)
        out[k] = v

    final = dict(requirements=requires)

    print('# This is a numina %s template file' % (__version__,))
    print('# for recipe %r' % (name,))
    print('#')
    if optional:
        print('# The following requirements are optional:')
        for kvals in optional.items():
            print('#  %s: %s' % kvals)
        print('# end of optional requirements')
    print(yaml.dump(final), end='')
    print('#products:')
    for prod in recipe.__provides__:
        print('# %s: %s' % print_io(prod))
    print('#logger:')
    print('# logfile: processing.log')
    print('# format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"')
    print('# enabled: true')
    print('---')

def print_recipe(recipe, name=None, insname=None, pipename=None, modename=None):
    try:
        if name is None:
            name = recipe.__module__ + '.' + recipe.__name__
        print('Recipe:', name)
        if recipe.__doc__:
            print(' summary:', recipe.__doc__.lstrip().expandtabs().splitlines()[0])
        if insname:
            print(' instrument:', insname)
        if pipename:
            print('  pipeline:', pipename)
        if modename:
            print('  obs mode:', modename)
        print(' requirements:')
        rp = RequirementParser(recipe)
        rp.print_requirements(pad='  ')
        print()
    except Exception as error:
        _logger.warning('problem %s with recipe %r', error, recipe)

def print_instrument(instrument, modes=True):
    print('Instrument:', instrument.name)
    for ic in instrument.configurations:
        print(' has configuration',repr(ic))
    for _, pl in instrument.pipelines.items():
        print(' has pipeline {0.name!r}, version {0.version}'.format(pl))
    if modes and instrument.modes:
        print(' has observing modes:')
        for mode in instrument.modes:
            print("  {0.name!r} ({0.key})".format(mode))

def print_obsmode(obsmode, instrument, ins=False):
    print('Observing Mode: {0.name!r} ({0.key})'.format(obsmode))
    print(' summary:', obsmode.summary)
    print(' instrument:', instrument.name)

def main(args=None):
    '''Entry point for the Numina CLI.'''

    # Configuration args from a text file
    config = configparser.SafeConfigParser()

    # Building programatically    
    config.add_section('numina')
    config.set('numina', 'format', 'yaml')

    # Custom values, site wide and local
    config.read(['.numina/numina.cfg',
                 os.path.join(xdg_config_home, 'numina/numina.cfg')])

    parser = argparse.ArgumentParser(
                 description='Command line interface of Numina',
                 prog='numina',
                 epilog="For detailed help pass --help to a target")
    
    parser.add_argument('-l', action="store", dest="logging", metavar="FILE",
                              help="FILE with logging configuration")
    parser.add_argument('-d', '--debug', action="store_true",
                          dest="debug", default=False,
                          help="make lots of noise")
    subparsers = parser.add_subparsers(title='Targets',
            description='These are valid commands you can ask numina to do.')

    parser_show_ins = subparsers.add_parser('show-instruments', 
            help='show registered instruments')

    parser_show_ins.set_defaults(command=show_instruments, verbose=0, what='om')
    parser_show_ins.add_argument('-o', '--observing-modes', 
                    action='store_true', dest='om', 
                    help='list observing modes of each instrument')
#    parser_show_ins.add_argument('--verbose', '-v', action='count')
    parser_show_ins.add_argument('name', nargs='*', default=None,
                             help='filter instruments by name')

    parser_show_mode = subparsers.add_parser('show-modes', 
            help='show information of observing modes')

    parser_show_mode.set_defaults(command=show_observingmodes, verbose=0, 
            what='om')
#    parser_show_mode.add_argument('--verbose', '-v', action='count')
    parser_show_mode.add_argument('-i','--instrument', 
                help='filter modes by instrument')
    parser_show_mode.add_argument('name', nargs='*', default=None,
                             help='filter observing modes by name')

    parser_show_rec = subparsers.add_parser('show-recipes', 
            help='show information of recipes')

    parser_show_rec.set_defaults(command=show_recipes, template=False)
    parser_show_rec.add_argument('-i','--instrument', 
                    help='filter recipes by instrument')
    parser_show_rec.add_argument('-t', '--template', action='store_true', 
                help='generate requirements YAML template')
#    parser_show_rec.add_argument('--output', type=argparse.FileType('wb', 0))
    parser_show_rec.add_argument('name', nargs='*', default=None,
                             help='filter recipes by name')

    parser_run = subparsers.add_parser('run', 
            help='process a observation result')
    
    parser_run.set_defaults(command=mode_run_obsmode)    

    parser_run.add_argument('-c', '--task-control', dest='reqs', 
            help='configuration file of the processing task', metavar='FILE')
    parser_run.add_argument('-r', '--requirements', dest='reqs', 
            help='alias for --task-control', metavar='FILE')
    parser_run.add_argument('-i','--instrument', dest='insconf', 
            default="default", help='name of an instrument configuration')
    parser_run.add_argument('-p','--pipeline', dest='pipe_name', 
        help='name of a pipeline')
    parser_run.add_argument('--basedir', action="store", dest="basedir", 
                      default=os.getcwd(),
        help='path to create the following directories')
    parser_run.add_argument('--datadir', action="store", dest="datadir", 
        help='path to directory containing pristine data')
    parser_run.add_argument('--resultsdir', action="store", dest="resultsdir",
        help='path to directory to store results')
    parser_run.add_argument('--workdir', action="store", dest="workdir",
        help='path to directory containing intermediate files')
    parser_run.add_argument('--cleanup', action="store_true", dest="cleanup", 
                    default=False, help='cleanup workdir on exit [disabled]')
    parser_run.add_argument('obsresult', 
            help='file with the observation result')

    parser_run_recipe = subparsers.add_parser('run-recipe', 
            help='run a recipe')
    
    parser_run_recipe.set_defaults(command=mode_run_recipe)    
    parser_run_recipe.add_argument('-c', '--task-control', dest='reqs', 
            help='configuration file of the processing task', metavar='FILE')
    parser_run_recipe.add_argument('-r', '--requirements', dest='reqs', 
            help='alias for --task-control', metavar='FILE')
    parser_run_recipe.add_argument('--obs-res', dest='obsresult', 
            help='observation result', metavar='FILE')
    parser_run_recipe.add_argument('recipe', 
            help='fqn of the recipe')
    parser_run_recipe.add_argument('--basedir', action="store", dest="basedir", 
                      default=os.getcwd(),
        help='path to create the following directories')
    parser_run_recipe.add_argument('--datadir', action="store", dest="datadir", 
        help='path to directory containing pristine data')
    parser_run_recipe.add_argument('--resultsdir', action="store", dest="resultsdir",
        help='path to directory to store results')
    parser_run_recipe.add_argument('--workdir', action="store", dest="workdir",
        help='path to directory containing intermediate files')
    parser_run_recipe.add_argument('--cleanup', action="store_true", dest="cleanup", 
                    default=False, help='cleanup workdir on exit [disabled]')
    
    args = parser.parse_args(args)

    # logger file
    if args.logging is not None:
        logging.config.fileConfig(args.logging)
    else:
        # This should be a default path in defaults.cfg
        try:
            args.logging = config.get('numina', 'logging')
            logging.config.fileConfig(args.logging)
        except configparser.Error:
            logging.config.dictConfig(_logconf)

    _logger = logging.getLogger("numina")
    _logger.info('Numina simple recipe runner version %s', __version__)

    args.drps = init_drp_system(namespace)

    args.command(args)

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
                _logger.warning('loading no validated instrument configuration')
                _logger.warning('you were warned')
            
            ins_conf = values.get('name', ins_conf)
            my_ins_conf = InstrumentConfiguration(ins_conf, values)

            # The new configuration must not overwrite existing configurations
            if ins_conf not in my_ins.configurations:
                my_ins.configurations[ins_conf] = my_ins_conf
            else:
                _logger.error('a configuration already exists %r, exiting', ins_conf)
            sys.exit(1)

        except IOError:
            _logger.error('instrument configuration %r does not exist', ins_conf)
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
        _logger.error('instrument %r does not have pipeline named %r', ins_name, pipe_name)
        sys.exit(1)

    _logger.info('loading pipeline %r', pipe_name)
    _logger.debug('pipeline object is %s', my_pipe)
    
    obs_mode = obsres.mode
    _logger.info("observing mode: %r", obs_mode)

    recipe_fqn = my_pipe.recipes.get(obs_mode)
    if recipe_fqn is None:
        _logger.error('pipeline %r does not have recipe to process %r obs mode', 
                            pipe_name, obs_mode)
        sys.exit(1)
    return recipe_fqn, pipe_name, my_ins_conf, ins_conf

def mode_run_obsmode(args):
    mode_run_common(args, mode='obs')

def mode_run_recipe(args):
    mode_run_common(args, mode='rec')

def mode_run_common(args, mode):
    '''Observing mode processing mode of numina.'''
    
    # Directories with relevant data
    workenv = WorkEnvironment(args.basedir, 
        workdir=args.workdir,
        resultsdir = args.resultsdir,    
        datadir = args.datadir)
        
    # Loading observation result if exists
    obsres = None
    if args.obsresult is not None:
        _logger.info("Loading observation result from %r", args.obsresult)
        
        with open(args.obsresult) as fd:
            obsres = obsres_from_dict(yaml.load(fd))
        
        _logger.debug('frames in observation result')
        for v in obsres.frames:
            _logger.debug('%r', v)

    if mode =='obs':
        recipe_fqn, pipe_name, my_ins_conf, ins_conf = load_from_obsres(obsres, args)
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
    logger_control = dict(logfile='processing.log',
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            enabled=True)
                
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
                    _logger.warning("'requirements' missing, using 'parameters'")
                    task_control_loaded['requirements'] = task_control_loaded['parameters']

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
    except ValidationError as error:
        _logger.error('%s, exiting', error)
        sys.exit(1)
    except RequirementError as error:
        _logger.error('%s, exiting', error)
        sys.exit(1)
    except (OSError, IOError) as exception:
        _logger.error('%s, exiting', exception)
        sys.exit(1)

    _logger.debug('parsing requirements')
    for i in recipeclass.__requires__:
        _logger.info("recipe requires %r", i.type.__class__)
        _logger.info("%r is %r", i.dest, getattr(rinput, i.dest))

    _logger.debug('parsing products')
    for req in recipeclass.__provides__:
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
        
    run_create_logger(recipe=recipe,
        task=task, rinput=rinput,
        workenv=workenv, task_control=task_control
        )

def run_create_logger(recipe, task, rinput, workenv, task_control):    
    '''Recipe execution mode of numina.'''
    
    # Creating custom logger file
    _recipe_logger_name = 'numina.recipes'
    _recipe_logger = logging.getLogger(_recipe_logger_name)
    logger_control = task_control['logger']
    if logger_control['enabled']:
        logfile = os.path.join(workenv.resultsdir, logger_control['logfile'])
        logformat = logger_control['format']
        fh = create_recipe_file_logger(_recipe_logger, logfile, logformat)
    else:
        fh = logging.NullHandler()

    _recipe_logger.addHandler(fh)

    try:
        csd = os.getcwd()
        _logger.debug('cwd to workdir')        
        os.chdir(workenv.workdir)
        task = internal_work(recipe, rinput, task)
        
        _logger.debug('cwd to resultdir: %r', workenv.resultsdir)
        os.chdir(workenv.resultsdir)
        
        _logger.info('storing result')
        
        guarda(task)
                    


        
    except StandardError as error:
        _logger.error('finishing with errors: %s', error)
    finally:
        _logger.debug('cwd to original path: %r', csd)
        os.chdir(csd)
        _recipe_logger.removeHandler(fh)


def internal_work(recipe, rinput, task):
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

def guarda(task):
    # Store results we know about
    # via store
    # for the rest dump with yaml
    
    
    result = task.result
    #result.suggest_store(**task_control['products'])
    saveres = {}
    for key in result.__stored__:
        val = getattr(result, key)
        store(val, 'disk')
        if hasattr(val, 'storage'):
            if val.storage['stored']:
                saveres[key] = val.storage['where']
        else:
            saveres[key] = val
    
    with open('result.txt', 'w+') as fd:
        yaml.dump(saveres, fd)

    # we put the results description here
    task.result = 'result.txt'

    # The rest goes here
    with open('task.txt', 'w+') as fd:
        yaml.dump(task.__dict__, fd)


if __name__ == '__main__':
    main()

