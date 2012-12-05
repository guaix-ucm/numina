#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

import os
import pkgutil
import logging
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
from numina.core import FrameDataProduct, BaseRecipe, DataProduct
from numina.core import BaseInstrument, InstrumentConfiguration
from numina.core.requirements import RequirementError
from numina.core.products import ValidationError
from numina.xdgdirs import xdg_config_home

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
                'level': 'INFO'
                 },
               'simple_console':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'simple',
                'level': 'INFO'
                 },
               'simple_console_warnings_only':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'simple',
                'level': 'WARNING'
                 },
               'detailed_console':
               {'class': 'logging.StreamHandler',                     
                'formatter': 'detailed',
                'level': 'INFO'
                 },
               },
  'loggers': {'numina': {'handlers': ['simple_console'], 'level': 'NOTSET', 'propagate': False},
              'numina.recipes': {'handlers': ['detailed_console'], 'level': 'NOTSET', 'propagate': False},
              },
  'root': {'handlers': ['detailed_console'], 'level': 'NOTSET'}
}

class ProcessingTask(object):
    pass

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

def super_load(path):
    spl = path.split('.')
    cls = spl[-1]
    mods = '.'.join(spl[:-1])
    import importlib
    mm = importlib.import_module(mods)
    Cls = getattr(mm, cls)
    return Cls

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
    for theins in instruments.values():
        if not args.instrument or (args.instrument == theins.name):
            for mode in theins.modes:
                if not args.name or (mode.recipe in args.name):
                    Cls = super_load(mode.recipe)
                    this_recipe_print(Cls, name=mode.recipe, insname=theins.name)

def show_observingmodes(args):
    for theins in instruments.values():
        if not args.instrument or (args.instrument == theins.name):
            for mode in theins.modes:
                if not args.name or (mode.key in args.name):
                    print_obsmode(mode, ins=True)

def show_instruments(args):
    for theins in instruments.values():
        if not args.name or (theins.name in args.name):
            print_instrument(theins, modes=args.om)

def print_recipe_template(recipe, name=None, insname=None):

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

def print_recipe(recipe, name=None, insname=None):
    try:
        if name:
            print('Recipe:', name)
        else:
            print('Recipe:', recipe.__module__ + '.' + recipe.__name__)
        if recipe.__doc__:
            print(' summary:', recipe.__doc__.lstrip().expandtabs().splitlines()[0])
        if insname:
            print(' instrument:', insname)
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
    for name, pl in instrument.pipelines.items():
        print(' has pipeline {0.name!r}, version {0.version}'.format(pl))
    if modes and instrument.modes:
        print(' has observing modes:')
        for mode in instrument.modes:
            print("  {0.name!r} ({0.key})".format(mode))

def print_obsmode(obsmode, ins=False):
    print('Observing Mode: {0.name!r} ({0.key})'.format(obsmode))
    print(' summary:', obsmode.summary)
    if ins:
        print(' instrument:', obsmode.instrument)
    print(' recipe:', obsmode.recipe)

def run_recipe(cls, obs):
    recipe = cls()
    result = recipe(obs)
    return result

def init_pipeline_system(namespace):
    '''Load all available pipelines from package 'namespace'.'''
    
    for imp, name, _is_pkg in pkgutil.walk_packages(namespace.__path__, namespace.__name__ + '.'):
        try:
            loader = imp.find_module(name)
            _mod = loader.load_module(name)
        except StandardError as error:
            _logger.warning('Problem importing %s, error of type %s with message "%s"', name, type(error), error)
        
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

    parser_show_ins = subparsers.add_parser('show-instruments', help='show registered instruments')

    parser_show_ins.set_defaults(command=show_instruments, verbose=0, what='om')
    parser_show_ins.add_argument('-o', '--observing-modes', 
                    action='store_true', dest='om', 
                    help='list observing modes of each instrument')
#    parser_show_ins.add_argument('--verbose', '-v', action='count')
    parser_show_ins.add_argument('name', nargs='*', default=None,
                             help='filter instruments by name')

    parser_show_mode = subparsers.add_parser('show-modes', help='show information of observing modes')

    parser_show_mode.set_defaults(command=show_observingmodes, verbose=0, what='om')
#    parser_show_mode.add_argument('--verbose', '-v', action='count')
    parser_show_mode.add_argument('-i','--instrument', help='filter modes by instrument')
    parser_show_mode.add_argument('name', nargs='*', default=None,
                             help='filter observing modes by name')

    parser_show_rec = subparsers.add_parser('show-recipes', help='show information of recipes')

    parser_show_rec.set_defaults(command=show_recipes, template=False)
    parser_show_rec.add_argument('-i','--instrument', 
                    help='filter recipes by instrument')
    parser_show_rec.add_argument('-t', '--template', action='store_true', 
                help='generate requirements YAML template')
#    parser_show_rec.add_argument('--output', type=argparse.FileType('wb', 0))
    parser_show_rec.add_argument('name', nargs='*', default=None,
                             help='filter recipes by name')

    parser_run = subparsers.add_parser('run', help='process a observation result')
    
    parser_run.set_defaults(command=mode_run)    

    parser_run.add_argument('-c', '--task-control', dest='reqs', help='configuration file of the processing task', metavar='FILE')
    parser_run.add_argument('-r', '--requirements', dest='reqs', help='alias for --task-control', metavar='FILE')
    parser_run.add_argument('-i','--instrument', dest='insconf', default="default", help='name of an instrument configuration')
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
    parser_run.add_argument('obsresult', help='file with the observation result')
    
    args = parser.parse_args(args)

    # logger file
    if args.logging is not None:
        logging.config.fileConfig(args.logging)
    else:
        # This should be a default path in defaults.cfg
        try:
            args.logging = config.get('numina', 'logging')
            logging.config.fileConfig(args.logging)
        except ConfigParserError:
            logging.config.dictConfig(_loggconf)

    _logger = logging.getLogger("numina")
    _logger.info('Numina simple recipe runner version %s', __version__)

    init_pipeline_system(namespace)

    global instruments
    instruments = {}

    _logger.debug('Loading instruments and pipelines')
    for InstrumentClass in BaseInstrument.__subclasses__():
        _logger.debug('Loading instrument %s', InstrumentClass.name)
        instruments[InstrumentClass.name] = InstrumentClass
        for key, conf in InstrumentClass.configurations.items():
            _logger.debug('with configuration %r', key)
        for key, pipe in InstrumentClass.pipelines.items():
            version = pipe.version
            _logger.debug('%s has pipeline %r, version %s', InstrumentClass.name, key, version)

    args.command(args)

def mode_run(args):
    '''Recipe execution mode of numina.'''
    obname = args.obsresult
    _logger.info("Loading observation result from %r", obname)

    with open(obname) as fd:
        obsres = obsres_from_dict(yaml.load(fd))

    _logger.info("Identifier of the observation result: %d", obsres.id)
    ins_name = obsres.instrument
    _logger.info("instrument name: %s", ins_name)
    MyInstrumentClass = instruments.get(ins_name)
    if MyInstrumentClass is None:
        _logger.error('instrument %r does not exist', ins_name)
        sys.exit(1)
    _logger.debug('instrument class is %s', MyInstrumentClass)

    if args.insconf is not None:
        _logger.debug("configuration from CLI is %r", args.insconf)
        ins_conf = args.insconf
    else:
        ins_conf = obsres.configuration

    _logger.info('loading instrument configuration %r', ins_conf)
    my_ins_conf = MyInstrumentClass.configurations.get(ins_conf)

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
            if ins_conf not in MyInstrumentClass.configurations:
                 MyInstrumentClass.configurations[ins_conf] = my_ins_conf
            else:
                _logger.error('a configuration already exists %r, exiting', ins_conf)
            sys.exit(1)

        except IOError:
            _logger.error('instrument configuration %r does not exist', ins_conf)
            sys.exit(1)

    my_ins = MyInstrumentClass(ins_conf)

    if args.pipe_name is not None:
        _logger.debug("pipeline from CLI is %r", args.pipe_name)
        pipe_name = args.pipe_name
    else:
        pipe_name = obsres.pipeline
        _logger.debug("pipeline from ObsResult is %r", pipe_name)

    my_pipe = MyInstrumentClass.pipelines.get(pipe_name)
    if my_pipe is None:
        _logger.error('instrument %r does not have pipeline named %r', ins_name, pipe_name)
        sys.exit(1)

    _logger.info('loading pipeline %r', pipe_name)
    _logger.debug('pipeline object is %s', my_pipe)
    
    obs_mode = obsres.mode
    _logger.info("observing mode: %r", obs_mode)

    MyRecipeClass = my_pipe.recipes.get(obs_mode)
    if MyRecipeClass is None:
        _logger.error('pipeline %r does not have recipe to process %r obs mode', 
                            pipe_name, obs_mode)
        sys.exit(1)
    _logger.debug('recipe class is %s', MyRecipeClass)

    logger_control = dict(logfile='processing.log',
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            enabled=True)
                
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

    for key in task_control:
        if key in task_control_loaded:
            task_control[key].update(task_control_loaded[key])

    _logger.debug('parsing requirements')
    rp = RequirementParser(MyRecipeClass)
    try:
        requires = rp.parse(task_control['requirements'], validate=False)
    except ValidationError as error:
        _logger.error('%s, exiting', error)
        sys.exit(1)
    except RequirementError as error:
        _logger.error('%s, exiting', error)
        sys.exit(1)

    for i in MyRecipeClass.__requires__:
        _logger.info("%r is %r", i.dest, getattr(requires, i.dest))

    _logger.debug('parsing products')
    for req in MyRecipeClass.__provides__:
        _logger.info('recipe provides %r', req)

    if logger_control['enabled']:
        _logger.debug('custom logger file: %s', logger_control['logfile'])
        _logger.debug('custom logger format: %s', logger_control['format'])
    else:
        _logger.debug('custom logger file disabled')

    _logger.debug('frames in observation result')
    for v in obsres.frames:
        _logger.debug('%r', v)

    if args.workdir is None:
        args.workdir = os.path.join(args.basedir, '_work')

    args.workdir = os.path.abspath(args.workdir)

    if args.resultsdir is None:
        args.resultsdir = os.path.join(args.basedir, '_results')

    args.resultsdir = os.path.abspath(args.resultsdir)

    if args.datadir is None:
        args.datadir = os.path.join(args.basedir, '_data')

    args.datadir = os.path.abspath(args.datadir)

    runinfo = {}
    runinfo['runner'] = 'numina'
    runinfo['runner_version'] = __version__
    runinfo['data_dir'] = args.datadir
    runinfo['work_dir'] = args.workdir
    runinfo['results_dir'] = args.resultsdir
    runinfo['pipeline'] = pipe_name

    _logger.debug('check datadir for pristine files: %r', runinfo['data_dir'])
    
    make_sure_path_doesnot_exist(runinfo['work_dir'])
    _logger.debug('check workdir for working: %r', runinfo['work_dir']) 
    make_sure_path_exists(runinfo['work_dir'])

    make_sure_path_doesnot_exist(runinfo['results_dir'])
    _logger.debug('check resultsdir to store results %r', runinfo['results_dir'])
    make_sure_path_exists(runinfo['results_dir'])

    _logger.info('copying files from %r to %r', runinfo['data_dir'], runinfo['work_dir'])
    try:
        _logger.debug('copying files from Observation Result')
        nframes = []
        for f in obsres.frames:
            complete = os.path.abspath(os.path.join(runinfo['data_dir'], f.filename))
            _logger.debug('copying %r to %r', f.filename, runinfo['work_dir'])
            shutil.copy(complete, runinfo['work_dir'])
        _logger.debug('copying files from Requirements')
        for i in MyRecipeClass.__requires__:
            if isinstance(i.type, FrameDataProduct):
                value = getattr(requires, i.dest)
                if value is not None:
                    _logger.debug('copying %r to %r', value.filename, runinfo['work_dir'])
                    complete = os.path.abspath(os.path.join(runinfo['data_dir'], value.filename))
                    shutil.copy(complete, runinfo['work_dir'])
    except (OSError, IOError) as exception:
        _logger.error('%s', exception)
        sys.exit(1)

    # Creating custom logger file
    _recipe_logger_name = 'numina.recipes'
    _recipe_logger = logging.getLogger(_recipe_logger_name)
    if logger_control['enabled']:
        logfile = os.path.join(runinfo['results_dir'], logger_control['logfile'])
        logformat = logger_control['format']
        fh = create_recipe_file_logger(_recipe_logger, logfile, logformat)
    else:
        fh = logging.NullHandler()

    _recipe_logger.addHandler(fh)

    # Running the recipe
    # we catch most exceptions
    try:
        TIMEFMT = '%FT%T'
        _logger.info('creating the recipe')
        recipe = MyRecipeClass()
        recipe.configure(instrument=my_ins)
        runinfo['recipe'] = recipe.__class__.__name__
        runinfo['recipe_full_name'] = fully_qualified_name(recipe.__class__)
        runinfo['recipe_version'] = recipe.__version__

        _logger.debug('cwd to workdir')
        csd = os.getcwd()
        task = ProcessingTask()
        os.chdir(runinfo['work_dir'])

        _logger.info('running recipe')
        now1 = datetime.datetime.now()
        runinfo['time_start'] = now1.strftime(TIMEFMT)
        result = recipe(obsres, requires)
        now2 = datetime.datetime.now()
        runinfo['time_end'] = now2.strftime(TIMEFMT)
        runinfo['time_running'] = now2 - now1
        _logger.info('result: %r', result)


        observation = {}
        observation['mode'] = obsres.mode
        observation['observing_result'] = obsres.id
        observation['instrument'] = obsres.instrument
        observation['instrument_configuration'] = ins_conf
        
        task.result = result
        task.runinfo = runinfo
        task.observation = observation
        
        # back to were we start
        os.chdir(csd)

        _logger.debug('cwd to resultdir: %r', runinfo['results_dir'])
        os.chdir(runinfo['results_dir'])
        _logger.info('storing result')

        result.suggest_store(**task_control['products'])

        with open('result.txt', 'w+') as fd:
            yaml.dump(task.__dict__, fd)
        
        _logger.debug('cwd to original path: %r', csd)
        os.chdir(csd)
    except StandardError as error:
        _logger.error('finishing with errors: %s', error)
    finally:
        _recipe_logger.removeHandler(fh)

if __name__ == '__main__':
    main()

