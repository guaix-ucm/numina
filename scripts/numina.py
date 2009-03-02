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
import ConfigParser
import os

import pyfits



version_number = "0.1"
version_line = '%prog ' + version_number
description = __doc__

def main():
    usage = "usage: %prog [options] recipe [recipe-options]"
    parser = OptionParser(usage=usage, version=version_line, description=description)
    parser.add_option('-d', '--debug',action="store_true", dest="debug", default=False, help="make lots of noise [default]")
    parser.add_option('-o', '--output',action="store", dest="filename", metavar="FILE", help="write output to FILE")
    parser.add_option('-l', action="store", dest="logging", metavar="FILE", help="FILE with logging configuration")
    # Stop when you find the first argument
    parser.disable_interspersed_args()
    (options, args) = parser.parse_args()
    
    # Configuration options from a text file    
    config = ConfigParser.SafeConfigParser()
    # Default values, it must exist
    config.readfp(open('defaults.cfg'))
    # Custom values, site wide and local
    config.read(['site.cfg', os.path.expanduser('~/.myapp.cfg')])

    loggini = config.get('ohoh', 'logging')

    logging.config.fileConfig(loggini)
        
    logger = logging.getLogger("numina")
    
#    if not options.debug:
#        l = logging.getLogger("emir")
#        l.setLevel(logging.INFO)
#        l = logging.getLogger("runner")
#        l.setLevel(logging.INFO)

        
    logger.debug('Numina EMIR recipe runner %s', version_number)
        
    filename='r%05d.fits'
    fits_conf = {'directory': 'images',
                 'index': 'images/index.pkl',
                 'filename': filename}
    
    from emir.simulation.storage import FitsStorage
    
    logger.info('Creating FITS storage')
    storage = FitsStorage(**fits_conf)
    
    # Registered actions:
    actions = {pyfits.HDUList: ('FITS file', storage.store)}
    
    default_module = config.get('ohoh', 'module')
        
    for i in args[0:1]:
        comps = i.split('.')
        recipe = comps[-1]
        logger.debug('recipe is %s',recipe)
        if len(comps) == 1:
            modulen = default_module
        else:
            modulen = '.'.join(comps[0:-1])
        logger.debug('module is %s', modulen)
    
        module = __import__(modulen)
        
        for part in modulen.split('.')[1:]:
            module = getattr(module, part)
            try:
                cons = getattr(module, recipe)
            except AttributeError:
                logger.error("% s doesn't exist", recipe)
                sys.exit(2)
    
        logger.debug('Created recipe instance')
        recipe = cons()
        # Parse the rest of the options
        logger.debug('Parsing the options provided by recipe instance')
        (options, noargs) = recipe.cmdoptions.parse_args(args=args[1:])
        
        for i in noargs:        
            logger.debug('Option file %s' % i)            
            recipe.iniconfig.read(i)
                    
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
