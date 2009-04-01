#
# Copyright 2008-2009 Sergio Pascual
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

# $Id$

'''User command line interface of Numina.

This is the long description.
It will span several lines'''


import logging
import logging.config
from optparse import OptionParser
from ConfigParser import SafeConfigParser
import os

import pyfits

from numina import class_loader, list_recipes
import numina.config as nconfig

__version__ = "$Revision$"

version_number = "0.0.1"
version_line = '%prog ' + version_number

def parse_cmdline(args=None):
    '''Parse the command line.'''
    usage = "usage: %prog [options] recipe [recipe-options]"
    
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
    parser.add_option('--list', action="store_const", const='list', 
                      dest="mode", default='none')
    parser.add_option('--run', action="store_const", const='run', 
                      dest="mode", default='none')
    
    # Stop when you find the first argument
    parser.disable_interspersed_args()
    (options, args) = parser.parse_args(args)
    return (options, args)

def mode_list(module):
    '''Run the list mode of Numina.'''
    list_recipes(module)
    
def mode_none():
    '''Do nothing in Numina.'''
    pass

def main(args=None):
    '''Entry point for the Numina CLI. '''        
    # Configuration options from a text file    
    config = SafeConfigParser()
    # Default values, it must exist
    config.readfp(open(os.path.join(os.path.dirname(__file__), 
                                    'defaults.cfg')))

    # Custom values, site wide and local
    config.read([os.path.join(nconfig.myconfigdir, 'site.cfg'), 
                 os.path.expanduser('~/.numina.cfg')])

    # The cmd line is parsed
    options, args = parse_cmdline(args)

    # After processing both the command line and the files
    # we get the values of everything

    # logger file
    loggini = os.path.join(nconfig.myconfigdir, 'logging.ini')

    logging.config.fileConfig(loggini)
        
    logger = logging.getLogger("numina")
    
    logger.info('Numina: EMIR recipe runner version %s', version_number)
        
    default_module = config.get('ohoh', 'module')
    
    if options.mode == 'list':
        mode_list(default_module)
        return 0
    elif options.mode == 'none':
        mode_none()
    
    # Here we are in mode run
    for rename in args[0:1]:
        try:
            RecipeClass = class_loader(rename, default_module, logger = logger)
        except AttributeError:
            return 2
        
        if RecipeClass is None:
            logger.error('%s is not a subclass of RecipeBase', rename)
            return 2
        
        # running the recipe
        logger.info('Created recipe instance of class %s', rename)
        recipe = RecipeClass()
        # Parse the rest of the options
        logger.debug('Parsing the options provided by recipe instance')
        (reoptions, reargs) = recipe.cmdoptions.parse_args(args=args[1:])
        
        if reoptions.docs:
            print rename, recipe.__doc__
            return 0
                    
        for i in reargs:        
            logger.debug('Option file %s' % i)            
            recipe.iniconfig.read(i)

        filename = 'r%05d.fits'
        fits_conf = {'directory': 'images',
                 'index': 'images/index.pkl',
                 'filename': filename}
    
        from emir.simulation.storage import FITSStorage
    
        logger.info('Creating FITS storage')
        storage = FITSStorage(**fits_conf)
        # Registered actions:
        actions = {pyfits.HDUList: ('FITS file', storage.store)}

                    
        logger.debug('Setting-up the recipe')
        recipe.setup()
        
        runs = recipe.repeat
        while not recipe.complete():
            logger.debug('Running the recipe instance %d of %d ', 
                         recipe.repeat, runs)
            result = recipe.run()
            logger.debug('Getting action for result')
            try:
                desc, action = actions[type(result)]
                logger.debug('Action for result is %s', desc)
                action(result)
            except KeyError:
                logger.warning('No action defined for type %s', type(result))
        
        logger.info('Completed execution')
        return 0

if __name__ == '__main__':
    main()
