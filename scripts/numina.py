#!/usr/bin/env python

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

'''This is a very long and convenient description of the program and its usage.

This is the long description.
It will span several lines'''


import logging
import logging.config
import sys
from optparse import OptionParser
from ConfigParser import SafeConfigParser
import os
import inspect

import pyfits

from emir.numina import Null
from emir.numina import class_loader
from emir.numina import list_recipes

version_number = "0.1"
version_line = '%prog ' + version_number
description = __doc__

def main():
    usage = "usage: %prog [options] recipe [recipe-options]"
    
    parser = OptionParser(usage=usage, version=version_line, description=description)
    # Command line options
    parser.add_option('-d', '--debug',action="store_true", dest="debug", default=False, help="make lots of noise [default]")
    parser.add_option('-o', '--output',action="store", dest="filename", metavar="FILE", help="write output to FILE")
    parser.add_option('-l', action="store", dest="logging", metavar="FILE", help="FILE with logging configuration")
    parser.add_option('--list', action="store_true", dest="listing", default=False)
        
    # Configuration options from a text file    
    config = SafeConfigParser()
    # Default values, it must exist
    config.readfp(open('defaults.cfg'))
    
    # Stop when you find the first argument
    parser.disable_interspersed_args()
    (options, args) = parser.parse_args()
        
    # Custom values, site wide and local
    config.read(['site.cfg', os.path.expanduser('~/.numina.cfg')])

    # After processing both the command line and the files
    # we get the values of everything

    # logger file
    loggini = config.get('ohoh', 'logging')

    logging.config.fileConfig(loggini)
        
    logger = logging.getLogger("numina")
    
    logger.info('Numina: EMIR recipe runner version %s', version_number)
        
    default_module = config.get('ohoh', 'module')
    
    if options.listing:
        list_recipes(default_module)
        sys.exit()
    
    for rename in args[0:1]:
        try:
            RecipeClass = class_loader(rename, default_module,  logger = logger)
        except AttributeError:
            sys.exit(2)
        
        if RecipeClass is None:
            logger.error('%s is not a subclass of RecipeBase', rename)
            sts.exit(2)
        
        # running the recipe
        logger.info('Created recipe instance of class %s', rename)
        recipe = RecipeClass()
        # Parse the rest of the options
        logger.debug('Parsing the options provided by recipe instance')
        (reoptions, reargs) = recipe.cmdoptions.parse_args(args=args[1:])
        
        if reoptions.docs:
            print rename, recipe.__doc__
            sys.exit(0)
                    
        for i in reargs:        
            logger.debug('Option file %s' % i)            
            recipe.iniconfig.read(i)

        filename='r%05d.fits'
        fits_conf = {'directory': 'images',
                 'index': 'images/index.pkl',
                 'filename': filename}
    
        from emir.simulation.storage import FitsStorage
    
        logger.info('Creating FITS storage')
        storage = FitsStorage(**fits_conf)
        # Registered actions:
        actions = {pyfits.HDUList: ('FITS file', storage.store)}

                    
        logger.debug('Setting-up the recipe')
        recipe.setup()
        
        runs = recipe.repeat
        while not recipe.complete():
            logger.debug('Running the recipe instance %d of %d ' % (recipe.repeat, runs))
            result = recipe.run()
            logger.debug('Getting action for result')
            try:
                desc, action = actions[type(result)]
                logger.debug('Action for result is %s', desc)
                action(result)
            except KeyError:
                logger.warning('No action defined for type %s', type(result))
        
        logger.info('Completed execution')

if __name__ == '__main__':
    main()
