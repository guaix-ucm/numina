#
# Copyright 2008-2010 Sergio Pascual
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

import uuid
import datetime
import logging.config
import os
from optparse import OptionParser
from ConfigParser import SafeConfigParser
import pkgutil
# get_data is not in python 2.5
if not hasattr(pkgutil, 'get_data'):
    from compatibility import get_data
    pkgutil.get_data = get_data
import StringIO

import xdg.BaseDirectory as xdgbd

from numina import __version__
from numina.logger import captureWarnings
from numina.recipes import list_recipes
from numina.recipes import init_recipe_system, list_recipes_by_obs_mode
from numina.diskstorage import store
import numina.recipes.registry as registry

def parse_cmdline(args=None):
    '''Parse the command line.'''
    usage = "usage: %prog [options] recipe [recipe-options]"

    version_line = '%prog ' + __version__ 

    parser = OptionParser(usage=usage, version=version_line, 
                          description=__doc__)
    # Command line options
    parser.add_option('-d', '--debug', action="store_true", 
                      dest="debug", default=False, 
                      help="make lots of noise [default]")
    parser.add_option('-o', '--output', action="store", dest="filename", 
                      metavar="FILE", help="write output to FILE")
    parser.add_option('-l', action="store", dest="logging", metavar="FILE", 
                      help="FILE with logging configuration")
    parser.add_option('--module', action="store", dest="module", 
                      metavar="FILE", help="FILE")
    parser.add_option('--list', action="store_const", const='list', 
                      dest="mode", default='none')
    parser.add_option('--run', action="store_const", const='run', 
                      dest="mode", default='none')
    
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

def mode_run(args, logger):
    
    registry.init_registry_from_file(args[0])

    try:
        instrument = registry.mget(['/observing_block/instrument',
                                    '/recipe/run/instrument'])

        obsmode = registry.mget(['/observing_block/mode', 
                                 '/recipe/run/mode'])
    except KeyError:
        logger.error('cannot retrieve instrument and obsmode')
        return 1
    
    logger.info('our instrument is %s and our observing mode is %s', 
                instrument, obsmode)
    
    for recipeClass in list_recipes():
        if instrument in recipeClass.instrument and obsmode in recipeClass.capabilities:
            parameters = {}
            par = registry.get('/recipe/parameters')
            for n, v, _ in recipeClass.required_parameters:
                
                # If the default value is ProxyPath
                # the default value is from the root configuration object
                #
                if isinstance(v, registry.ProxyPath):
                    try:
                        defval = v.get()
                    except KeyError:
                        defval = None
                else:
                    defval = v
                parameters[n] = par.get(n, defval)
                logger.debug('parameter %s = %s',n, parameters[n])
                
                
            logger.debug('Creating the recipe')
            runinfo = registry.get('/recipe/run')
            recipe = recipeClass(parameters, runinfo)
            
            errorcount = 0
    
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
                uuidstr = str(uuid.uuid1())
                
                store(result, 'numina-%s-%s.log' % (nowstr, uuidstr))
                
            logger.debug('Cleaning up the recipe')
            recipe.cleanup()
            
            if errorcount > 0:
                logger.error('Errors during execution: %d', errorcount)
                return errorcount
            else:
                logger.info('Completed execution')

            return 0
    else:
        logger.error('observing mode %s is not processed by any recipe', obsmode)
        
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
   
    config.readfp(StringIO.StringIO(pkgutil.get_data('numina','defaults.cfg')))

    # Custom values, site wide and local
    config.read([os.path.join(xdgbd.xdg_config_home, 'numina/numina.cfg')])
    
    # The cmd line is parsed
    options, args = parse_cmdline(args)

    # After processing both the command line and the files
    # we get the values of everything

    # logger file
    if options.logging is None:
        options.logging = StringIO.StringIO(pkgutil.get_data('numina','logging.ini'))

    logging.config.fileConfig(options.logging)
    
    logger = logging.getLogger("numina")
    
    logger.info('Numina: EMIR recipe runner version %s', __version__)
    
    if options.module is None:
        options.module = config.get('ohoh', 'module')
    
    init_recipe_system([options.module])
    captureWarnings(True)
    
    if options.mode == 'list':
        mode_list() 
        return 0
    elif options.mode == 'none':
        mode_none()
        return 0
    elif options.mode == 'run':
        return mode_run(args, logger)
    
if __name__ == '__main__':
    main()
