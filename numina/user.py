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

import logging.config
import os
import argparse
from ConfigParser import SafeConfigParser
from ConfigParser import Error as ConfigParserError
from logging import captureWarnings
import inspect
import traceback
import shutil

from numina.treedict import TreeDict
from numina import __version__, obsres_from_dict
from numina.pipeline import init_pipeline_system
from numina.recipes import list_recipes
from numina.pipeline import get_recipe
from numina.serialize import lookup
from numina.xdgdirs import xdg_config_home

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

def mode_list(serializer, args):
    '''Run the list mode of Numina'''
    _logger.debug('list mode')
    for recipeCls in list_recipes():
#        print fully_qualified_name(recipeCls)
        print recipeCls

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
        _logger.warning('cannot find entry point for %(instrument)s and %(mode)s', obsres.__dict__)
        raise

    _logger.info('matching parameters')    

    parameters = {}

    for req in RecipeClass.__requires__:
        try:
            _logger.info('recipe requires %s', req.name)
            parameters[req.name]= req.lookup(params)
            _logger.debug('parameter %s has value %s', req.name, parameters[req.name])
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

def run_recipe(serializer, obsres, params, instrument, workdir, resultsdir, cleanup): 
    _logger.info('instrument={0.instrument} mode={0.mode}'.format(obsres))
    _logger.info('pipeline={0[pipeline]}'.format(instrument))
    try:
        RecipeClass = get_recipe(instrument['pipeline'], obsres.mode)
        _logger.info('entry point is %s %d', RecipeClass, id(RecipeClass))
    except ValueError:
        _logger.warning('cannot find recipe class for {0.instrument} mode={0.mode}'
                        .format(obsres))
        raise

    _logger.info('matching parameters')    

    allmetadata = params
    allmetadata['instrument'] = instrument
    allm = TreeDict(allmetadata)
    parameters = {}

    for req in RecipeClass.__requires__:
        try:
            _logger.info('recipe requires %s', req.name)
            parameters[req.name]= req.lookup(allm)
            _logger.debug('parameter %s has value %s', req.name, parameters[req.name])
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
    runinfo['entrypoint'] = fully_qualified_name(RecipeClass)

    # Set custom logger
    _logger.debug('getting recipe logger')
    
    _recipe_logger = RecipeClass.logger
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

        result = main_internal(RecipeClass, obsres, instrument, parameters, 
                                runinfo, workdir=workdir)

        result['recipe_runner'] = info()
        result['runinfo'] = runinfo
    
        os.chdir(resultsdir)

        with open('result.txt', 'w+') as fd:
            serializer.dump(result, fd)
    
        with open('result.txt', 'r') as fd:
            result = serializer.load(fd)

        if cleanup:
            _logger.debug('cleaning up the workdir')
            shutil.rmtree(workdir)
        _logger.info('finished')
    except Exception as error:
        _logger.error('%s', error)
        result['error'] = {'type': error.__class__.__name__, 
                                    'message': str(error), 
                                    'traceback': traceback.format_exc()}
        _logger.error('finishing with errors: %s', error)
    finally:
        _recipe_logger.removeHandler(fh)

    return result



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
    if args.obsblock is not None:
        _logger.info('reading observing block from %s', args.obsblock)
        with open(args.obsblock, 'r') as fd:
            obsres_read = True
            obsres_dict = serializer.load(fd)
            obsres = obsres_from_dict(obsres_dict)
    
    # Read instrument information from args.instrument
    if args.instrument is not None:
        _logger.info('reading instrument config from %s', args.instrument)
        with open(args.instrument, 'r') as fd:
            # json decode    
            instrument = serializer.load(fd)
            instrument_read = True

    # Read task information from args.task
    with open(args.task, 'r') as fd:
        task_control = serializer.load(fd)
            
    if not instrument_read and 'instrument' in task_control:
        _logger.info('reading instrument config from %s', args.task)
        instrument = task_control['instrument']
        
    if not obsres_read and 'observing_result' in task_control:
        _logger.info('reading observing result from %s', args.task)
        obsres_read = True
        obsres_dict = task_control['observing_result']
        obsres = obsres_from_dict(obsres_dict)

    if 'parameters' in task_control:
        _logger.info('reading reduction parameters from %s', args.task)
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
    parser.add_argument('--module', action="store", dest="module", 
                      metavar="FILE", help="FILE", default='emir')
    
    subparsers = parser.add_subparsers(title='Targets',
                                       description='These are valid commands you can ask numina to do.')
    parser_list = subparsers.add_parser('list', help='list help')
    
    parser_list.set_defaults(command=mode_list)
    
    parser_run = subparsers.add_parser('run', help='run help')
    
    parser_run.set_defaults(command=mode_run)    

    parser_run.add_argument('--instrument', dest='instrument', default=None)
    parser_run.add_argument('--obsblock', dest='obsblock', default=None)
    parser_run.add_argument('--basedir', action="store", dest="basedir", 
                      default=os.getcwd())
    # FIXME: It is questionable if this flag should be used or not
    parser_run.add_argument('--datadir', action="store", dest="datadir")
    parser_run.add_argument('--resultsdir', action="store", dest="resultsdir")
    parser_run.add_argument('--workdir', action="store", dest="workdir")    
    parser_run.add_argument('--cleanup', action="store_true", dest="cleanup", 
                      default=False)
    parser_run.add_argument('task')
    
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

    pipelines = init_pipeline_system()
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
