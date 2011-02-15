#
# Copyright 2008-2011 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

'''User command line interface of Numina.'''

import datetime
import logging.config
import os
from optparse import OptionParser
from ConfigParser import SafeConfigParser
from ConfigParser import Error as CPError

from compatibility import get_data

import StringIO

import xdg.BaseDirectory as xdgbd

from numina import __version__
from numina.logger import captureWarnings
from numina.recipes import list_recipes
from numina.recipes import init_recipe_system
from numina.diskstorage import store
import numina.recipes.registry as registry

def parse_cmdline(args=None):
    '''Parse the command line.'''
    usage = "usage: %prog [options] recipe [recipe-options]"

    version_line = '%prog ' + __version__ 

    parser = OptionParser(usage=usage, version=version_line, 
                          description=__doc__)
    # Command line options
    parser.set_defaults(mode="none")
    parser.add_option('-d', '--debug', action="store_true", 
                      dest="debug", default=False, 
                      help="make lots of noise [default]")
    parser.add_option('-l', action="store", dest="logging", metavar="FILE", 
                      help="FILE with logging configuration")
    parser.add_option('--module', action="store", dest="module", 
                      metavar="FILE", help="FILE")
    parser.add_option('--list', action="store_const", const='list', 
                      dest="mode")
    parser.add_option('--run', action="store_const", const='run', 
                      dest="mode")
    parser.add_option('--basedir', action="store", dest="basedir", 
                      default=os.getcwd())
    parser.add_option('--resultsdir', action="store", dest="resultsdir")
    parser.add_option('--workdir', action="store", dest="workdir")
    parser.add_option('--datadir', action="store", dest="datadir")
    
    parser.add_option('--cleanup', action="store_true", dest="cleanup", 
                      default=False)
    # Stop when you find the first argument
    parser.disable_interspersed_args()
    (options, args) = parser.parse_args(args)
    return (options, args)

def mode_list():
    '''Run the list mode of Numina'''
    for recipeclass in list_recipes():
        print recipeclass
    
def mode_none():
    '''Do nothing in Numina.'''
    pass

def mode_run(args, logger, options):
    
    registry.init_registry_from_file(args[0])

    try:
        instrument = registry.get('/observing_block/instrument')
        obsmode = registry.get('/observing_block/mode')
    except KeyError:
        logger.error('cannot retrieve instrument and obsmode')
        return 1
    
    logger.info('our instrument is %s and our observing mode is %s', 
                instrument, obsmode)
    
    nrecipes = 0
    
    # Creating base directory for storing results
    
    workdir = options.workdir
    datadir = options.datadir
    resultsdir = options.resultsdir
    
    for recipeClass in list_recipes():
        if (instrument in recipeClass.instrument 
            and obsmode in recipeClass.capabilities 
            and '__main__' != recipeClass.__module__):
            logger.info('Recipe is %s', recipeClass)
            nrecipes += 1
            parameters = {}
            
            fullname = "%s.%s" % (recipeClass.__module__,  recipeClass.__name__)
            
            par = registry.mget(['/recipes/default/parameters',
                                 '/recipes/%s/parameters' % fullname])
            for n, v, _ in recipeClass.required_parameters:
                
                # If the default value is ProxyPath
                # the default value is from the root configuration object
                #
                if isinstance(v, (registry.ProxyPath, registry.ProxyQuery)):
                    try:
                        defval = v.get()
                    except KeyError:
                        defval = None
                else:
                    defval = v
                parameters[n] = par.get(n, defval)
                logger.debug('parameter %s = %s',n, parameters[n])
            
            # Default runinfo
            runinfo = dict(nthreads=1)
             
            logger.debug('Creating the recipe')
            runinfo.update(registry.mget(['/recipes/%s/run' % fullname, 
                                     '/recipes/default/run']))
            
            runinfo['workdir'] = workdir
            runinfo['datadir'] = datadir
            runinfo['resultsdir'] = resultsdir
            
            recipe = recipeClass(parameters, runinfo)
            
            os.chdir(datadir)
            recipe.setup()
            
            errorcount = 0
    
            # Running the recipe
            os.chdir(workdir)
            for result in recipe():
                logger.info('Running the recipe instance %d of %d ', 
                         recipe.current + 1, recipe.repeat)
                
                result['recipe_runner'] = info()
                result['instrument'] = {'name': instrument, 
                                        'mode': obsmode}
                
                if result['run']['status'] != 0:
                    errorcount += 1
                
                # Creating the filename for the result
                nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                rdir = '%s-%d' % (fullname, recipe.current + 1)
                
                storedir = os.path.join(resultsdir, rdir)
                os.mkdir(storedir)
                os.chdir(storedir)
                store(result, 'numina-%s.log' % nowstr)
                os.chdir(workdir)
                
            logger.debug('Cleaning up the recipe')
            recipe.cleanup()
            
            if errorcount > 0:
                logger.error('Errors during execution: %d', errorcount)
            else:
                logger.info('Completed execution')

            #return 0
    if nrecipes == 0:
        logger.error('observing mode %s is not processed by any recipe', obsmode)

    
    import shutil
    if options.cleanup:
        logger.debug('Cleaning up the workdir')
        shutil.rmtree(workdir)
        
    return 0

def info():
    '''Information about this version of numina.
    
    This information will be stored in the result object of the recipe
    '''
    return dict(name='numina', version=__version__)

def main(args=None):
    '''Entry point for the Numina CLI.'''        
    # Configuration options from a text file    
    config = SafeConfigParser()
    # Default values, it must exist
   
    config.readfp(StringIO.StringIO(get_data('numina','defaults.cfg')))

    # Custom values, site wide and local
    config.read(['.numina/numina.cfg', 
                 os.path.join(xdgbd.xdg_config_home, 'numina/numina.cfg')])
    
    # The cmd line is parsed
    options, args = parse_cmdline(args)

    # After processing both the command line and the files
    # we get the values of everything

    # logger file
    if options.logging is None:
        # This should be a default path in defaults.cfg
        try:
            options.logging = config.get('numina', 'logging')
        except CPError:
            options.logging = StringIO.StringIO(get_data('numina','logging.ini'))

    logging.config.fileConfig(options.logging)
    
    logger = logging.getLogger("numina")
    
    logger.info('Numina simple recipe runner version %s', __version__)
    
    if options.module is None:
        options.module = config.get('numina', 'module')
    
    init_recipe_system([options.module])
    captureWarnings(True)
    
    if options.basedir is None:
        options.basedir = os.getcwd()
    
    if options.workdir is None:
        options.workdir = os.path.join(options.basedir, 'work')
    else:
        options.workdir = os.path.abspath(options.workdir)
        
    if options.datadir is None:
        options.datadir = os.path.join(options.basedir, 'data')
    else:
        options.datadir = os.path.abspath(options.datadir)
        
    if options.resultsdir is None:
        options.resultsdir = os.path.join(options.basedir, 'results')
    else:
        options.resultsdir = os.path.abspath(options.resultsdir)
    
    if options.mode == 'list':
        mode_list() 
        return 0
    elif options.mode == 'none':
        mode_none()
        return 0
    elif options.mode == 'run':
        
        # Check workdir exists
        if not os.path.exists(options.workdir):
            os.mkdir(options.workdir)
        # Check resultdir exists
        if not os.path.exists(options.resultsdir):
            os.mkdir(options.resultsdir)
        
        return mode_run(args, logger, options)
    
if __name__ == '__main__':
    main()
