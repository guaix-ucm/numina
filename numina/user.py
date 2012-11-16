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

import sys
import logging.config
import os
import argparse
from ConfigParser import SafeConfigParser
from ConfigParser import Error as ConfigParserError
from logging import captureWarnings
import inspect
import traceback
import shutil

from numina import __version__
from numina.core import obsres_from_dict
from numina.core import init_pipeline_system
from numina.core import list_recipes
from numina.core import RequirementParser
from numina.core import get_recipe, get_instruments
from numina.serialize import lookup
from numina.xdgdirs import xdg_config_home
from numina.treedict import TreeDict

_logger = logging.getLogger("numina")

_loggconf = {'version': 1,
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

def fully_qualified_name(obj, sep='.'):
    if inspect.isclass(obj):
        return obj.__module__ + sep + obj.__name__
    else:
        return obj.__module__ + sep + obj.__class__.__name__

def super_load(path):
    spl = path.split('.')
    cls = spl[-1]
    mods = '.'.join(spl[:-1])
    import importlib
    mm = importlib.import_module(mods)
    Cls = getattr(mm, cls)
    return Cls
    
    
def mode_show(serializer, args):
    '''Run the show mode of Numina'''
    _logger.debug('show mode')
    if args.what == 'om':
        show_observingmodes(args)
    elif args.what == 'rcp':
        show_recipes(args)
    else:
        show_instruments(args)
        
def show_recipes(args):
    if args.id is None:
        for Cls in list_recipes():
            print_recipe(Cls)            
    else:
        Cls = super_load(args.id)
        print_recipe(Cls)    

def show_observingmodes(args):
    
    if args.id:
        must_print = lambda x: x.uuid == args.id 
    else:
        must_print = lambda x: True
    
    ins = get_instruments()
    for theins in ins.values():
        for mode in theins.modes:
            if must_print(mode):
                print_obsmode(mode)

def show_instruments(args):
    ins = get_instruments()
    for theins in ins.values():
        print_instrument(theins, modes=False)

def print_recipe(recipe):
    print('Recipe:', recipe.__module__ + '.' + recipe.__name__)
    if recipe.__doc__:
        print('Summary:', recipe.__doc__.expandtabs().splitlines()[0])
    print('Requirements')
    print('------------')
    rp = RequirementParser(recipe.__requires__)
    rp.print_requirements()
    print()

def print_instrument(instrument, modes=True):
    print('Name:', instrument.name)
    
    if modes and instrument.modes:
        print('Observing modes')
        print('---------------')
        for mode in instrument.modes:
            print_obsmode(mode)
        print('---')

def print_obsmode(obsmode):
    print('%s: %s' % (obsmode.name, obsmode.summary))
    print('Instrument:', obsmode.instrument)
    print('Recipe:', obsmode.recipe)
    print('Key:', obsmode.key)
    print('UUID:', obsmode.uuid)
    print('--')

def main_internal(cls, obsres, 
    instrument, 
    parameters, 
    runinfo, 
    workdir=None):

    csd = os.getcwd()

    if workdir is not None:
        workdir = os.path.abspath(workdir)

    recipe = cls()

    recipe.configure(instrument=instrument,
                    parameters=parameters,
                    runinfo=runinfo)

    os.chdir(workdir)
    try:
        result = recipe(obsres)
    finally:
        os.chdir(csd)

    return result

# part of this code appears in
# pontifex/process.py

def run_recipe_from_file(serializer, task_control, workdir=None, resultsdir=None, cleanup=False):

    workdir = os.getcwd() if workdir is None else workdir
    resultsdir = os.getcwd if resultsdir is None else resultsdir

    # json decode
    with open(task_control, 'r') as fd:
        task_control = serializer.load(fd)
    
    ins_pars = {}
    params = {}
    
    if 'instrument' in task_control:
        _logger.info('file contains instrument config')
        ins_pars = task_control['instrument']
    if 'observing_result' in task_control:
        _logger.info('file contains observing result')
        obsres_dict = task_control['observing_result']
        obsres = obsres_from_dict(obsres_dict)

    if 'reduction' in task_control:
        params = task_control['reduction']['parameters']
        
    _logger.info('instrument=%(instrument)s mode=%(mode)s', 
                obsres.__dict__)
    _logger.info('pipeline=%s', ins_pars['pipeline'])
    try:
        RecipeClass = get_recipe(ins_pars['pipeline'], obsres.mode)
        _logger.info('entry point is %s', RecipeClass)
    except ValueError:
        _logger.error('cannot find entry point for %(instrument)s and %(mode)s', obsres.__dict__)
        sys.exit(1)

    _logger.info('matching parameters')    

    parameters = {}
    reqparser = RequirementParser(RecipeClass.__requires__)

    try:
        parameters = reqparser.parse(params)
        names = reqparser.parse2(params)
    except LookupError as error:
        _logger.error('%s', error)
        raise

    for req in RecipeClass.__provides__:
        _logger.info('recipe provides %s', req)
    
    # Creating base directory for storing results
                   
    _logger.debug('creating runinfo')
            
    runinfo = {}
    runinfo['workdir'] = workdir
    runinfo['resultsdir'] = resultsdir
    runinfo['entrypoint'] = RecipeClass

    # Set custom logger
    # FIXME we are assuming here that Recipe top package is named after the instrument
    _recipe_logger = logging.getLogger('%(instrument)s.recipes' % obsres.__dict__)
    _recipe_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    _logger.debug('creating custom logger "processing.log"')
    os.chdir(resultsdir)
    fh = logging.FileHandler('processing.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_recipe_formatter)
    _recipe_logger.addHandler(fh)

    result = {}
    try:
        # Running the recipe
        _logger.debug('running the recipe %s', RecipeClass.__name__)

        result = main_internal(RecipeClass, obsres, ins_pars, parameters, 
                                runinfo, workdir=workdir)

        result['recipe_runner'] = info()
        result['runinfo'] = runinfo
    
        os.chdir(resultsdir)

        with open('result.txt', 'w+') as fd:
            serializer.dump(result, fd)
    
        with open('result.txt', 'r') as fd:
            result = serializer.load(fd)

        if cleanup:
            _logger.debug('Cleaning up the workdir')
            shutil.rmtree(workdir)
    except Exception as error:
        _logger.error('%s', error)
        result['error'] = {'type': error.__class__.__name__, 
                                    'message': str(error), 
                                    'traceback': traceback.format_exc()}
    finally:
        _recipe_logger.removeHandler(fh)

    return result

class TaskResult(object):
    pass

def run_recipe(serializer, obsres, params, instrument, workdir, resultsdir, cleanup): 
    _logger.info('instrument={0.instrument} mode={0.mode}'.format(obsres))
    _logger.info('pipeline={0[pipeline]}'.format(instrument))
    try:
        RecipeClass = get_recipe(instrument['pipeline'], obsres.mode)
        _logger.info('entry point is %s', RecipeClass)
    except ValueError:
        _logger.error('cannot find recipe class for {0.instrument} mode={0.mode}'
                        .format(obsres))
        sys.exit(1)

    _logger.info('matching parameters')    

    allmetadata = params
    allmetadata['instrument'] = instrument
    allm = TreeDict(allmetadata)
    parameters = {}

    reqparser = RequirementParser(RecipeClass.__requires__)

    try:
        parameters = reqparser.parse(allm)
    except LookupError as error:
        _logger.error('%s', error)
        sys.exit(1)

    for req in RecipeClass.__provides__:
        _logger.info('recipe provides %s', req)
    
    # Creating base directory for storing results
                   
    _logger.debug('creating runinfo')
            
    runinfo = {}
    runinfo['workdir'] = workdir
    runinfo['resultsdir'] = resultsdir
    runinfo['entrypoint'] = fully_qualified_name(RecipeClass)

    # Set custom logger
    _logger.debug('getting recipe logger')
    
    _recipe_logger = RecipeClass.logger
    _recipe_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logfile = 'processing.log'
    _logger.debug('creating custom logger "%s"', logfile)
    os.chdir(resultsdir)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_recipe_formatter)
    _recipe_logger.addHandler(fh)

    task = TaskResult()
    try:
        # Running the recipe
        _logger.debug('running the recipe %s', RecipeClass.__name__)

        result = main_internal(RecipeClass, obsres, instrument, parameters, 
                                runinfo, workdir=workdir)

        task.result = result
        task.recipe_runner = info()
        task.runinfo = runinfo
    
        os.chdir(resultsdir)

        with open('result.txt', 'w+') as fd:
            serializer.dump(task, fd)
    
        with open('result.txt', 'r') as fd:
            task = serializer.load(fd)

        if cleanup:
            _logger.debug('cleaning up the workdir')
            shutil.rmtree(workdir)
        _logger.info('finished')
    except Exception as error:
        _logger.error('%s', error)
        task.error = error
        _logger.error('finishing with errors: %s', error)
    finally:
        _recipe_logger.removeHandler(fh)

    return task



def mode_run(serializer, args):
    '''Recipe execution mode of numina.'''
    args.basedir = os.path.abspath(args.basedir)    
    
    if args.workdir is None:
        args.workdir = os.path.join(args.basedir, 'work')
    
    args.workdir = os.path.abspath(args.workdir)
                
    if args.resultsdir is None:
        args.resultsdir = os.path.join(args.basedir, 'results')
    
    args.resultsdir = os.path.abspath(args.resultsdir)

    # Check basedir exists
    if not os.path.exists(args.basedir):
        os.mkdir(args.basedir)
        
    # Check workdir exists
    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    
    # Check resultdir exists
    if not os.path.exists(args.resultsdir):
        os.mkdir(args.resultsdir)

    instrument = {}
    reduction = {}

    obsres_read = False
    instrument_read = False
    
    # Read observing result from args.
    _logger.info('reading observing block from %s', args.obsblock)
    with open(args.obsblock, 'r') as fd:
        obsres_read = True
        obsres_dict = serializer.load(fd)
        obsres = obsres_from_dict(obsres_dict)
    
    # Read instrument information from args.instrument
    if args.instrument is not None:
        _logger.info('reading instrument config from %s', args.instrument)
        with open(args.instrument, 'r') as fd:
            instrument = serializer.load(fd)
            instrument_read = True

    # Read task information from args.task
    if args.reqs is not None:
        with open(args.reqs, 'r') as fd:
            task_control = serializer.load(fd)
    else:
        task_control = {}
            
    if not instrument_read and 'instrument' in task_control:
        _logger.info('reading instrument config from %s', args.task)
        instrument = task_control['instrument']
        
    if not obsres_read and 'observing_result' in task_control:
        _logger.info('reading observing result from %s', args.reqs)
        obsres_read = True
        obsres_dict = task_control['observing_result']
        obsres = obsres_from_dict(obsres_dict)

    if 'parameters' in task_control:
        _logger.info('reading reduction parameters from %s', args.reqs)
        reduction = task_control['parameters']
        
    if not obsres_read:
        _logger.error('observing result not read from input files')
        return
        
    if not instrument_read:
        _logger.error('instrument not read from input files')
        return
    
    return run_recipe(serializer, obsres, reduction, instrument, 
                       args.workdir, args.resultsdir, args.cleanup)

def info():
    '''Information about this version of numina.
    
    This information will be stored in the result object of the recipe
    '''
    return dict(name='numina', version=__version__)

def main(args=None):
    '''Entry point for the Numina CLI.'''        
    # Configuration args from a text file    
    config = SafeConfigParser()

    # Building programatically    
    config.add_section('numina')
    config.set('numina', 'format', 'yaml')

    # Custom values, site wide and local
    config.read(['.numina/numina.cfg', 
                 os.path.join(xdg_config_home, 'numina/numina.cfg')])

    parser = argparse.ArgumentParser(description='Command line interface of Numina',
                                     prog='numina',
                                     epilog="For detailed help pass " \
                                               "--help to a target")
    
    parser.add_argument('-d', '--debug', action="store_true", 
                      dest="debug", default=False, 
                      help="make lots of noise [default]")
    parser.add_argument('-l', action="store", dest="logging", metavar="FILE", 
                      help="FILE with logging configuration")

    subparsers = parser.add_subparsers(title='Targets',
                                       description='These are valid commands you can ask numina to do.')
    parser_show = subparsers.add_parser('show', help='show help')
    
    parser_show.add_argument('-o', '--observing-modes', action='store_const', dest='what', const='om',
                             help='Show observing modes')
    parser_show.add_argument('-r', action='store_const', dest='what', const='rcp',
                             help='Show recipes')
    parser_show.add_argument('-i', action='store_const', dest='what', const='ins',
                             help='Show instruments')
    
    parser_show.set_defaults(command=mode_show, what='om')
    parser_show.add_argument('id', nargs='?', default=None,
                             help='Identificator')
    
    parser_run = subparsers.add_parser('run', help='run help')
    
    parser_run.set_defaults(command=mode_run)    

    parser_run.add_argument('--instrument', dest='instrument', default=None)
    parser_run.add_argument('--parameters', dest='reqs', default=None)
    parser_run.add_argument('--requirements', dest='reqs', default=None)
    parser_run.add_argument('--basedir', action="store", dest="basedir", 
                      default=os.getcwd())
    # FIXME: It is questionable if this flag should be used or not
    parser_run.add_argument('--datadir', action="store", dest="datadir")
    parser_run.add_argument('--resultsdir', action="store", dest="resultsdir")
    parser_run.add_argument('--workdir', action="store", dest="workdir")    
    parser_run.add_argument('--cleanup', action="store_true", dest="cleanup", 
                      default=False)
    parser_run.add_argument('obsblock')
    
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

    instruments, pipelines = init_pipeline_system()
    for key in instruments:
        pl = instruments[key]
        name = pl.name
        _logger.info('Loaded instrument %s, %s', key, name)

    for key in pipelines:
        pl = pipelines[key]
        version = pl.version
        instrument = key
        _logger.info('Loaded pipeline for %s, version %s', instrument, version)
    captureWarnings(True)

    # Serialization loaded after pipelines, so that
    # pipelines can provide their own
    # serialization format
    serformat = config.get('numina', 'format')
    _logger.info('Serialization format is %s', serformat)
    try:
        serializer = lookup(serformat)      
    except LookupError:
        _logger.info('Serialization format %s is not defined', serformat)
        raise
    
    args.command(serializer, args)

if __name__ == '__main__':
    main()
