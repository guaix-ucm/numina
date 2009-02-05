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

import logging

logger = logging.getLogger("emir")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

logger = logging.getLogger("runner")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

import sys
from optparse import OptionParser

version_number = "0.1"

def main():
    usage = "usage: %prog [options] recipe [recipe-options]"
    description='''this is a very long and convenient description of the programa and its usage
    it will span several lines'''
    parser = OptionParser(usage=usage, version="%prog 0.1", description=description)
    parser.add_option('-d', '--debug',action="store_true", dest="debug", default=False, help="make lots of noise [default]")
    parser.add_option('-o', '--output',action="store", dest="filename", metavar="FILE", help="write output to FILE")
    # Stop when you find the first argument
    parser.disable_interspersed_args()
    (options, args) = parser.parse_args()
    
      
    
    if not options.debug:
        l = logging.getLogger("emir")
        l.setLevel(logging.INFO)
        l = logging.getLogger("runner")
        l.setLevel(logging.INFO)
        
    logger.debug('Emir recipe runner %s', version_number) 
        
    for i in args[0:1]:
        comps = i.split('.')
        recipe = comps[-1]
        logger.debug('recipe is %s',recipe)
        if len(comps) == 1:
            modulen = 'emir.recipes'
        else:
            modulen = '.'.join(comps[0:-1])
        logger.debug('module is %s', modulen)
    
        module = __import__(modulen)
        
        for part in modulen.split('.')[1:]:
            module = getattr(module, part)
            try:
                cons = getattr(module,recipe)
            except AttributeError:
                logger.error("% s doesn't exist", recipe)
                sys.exit(2)
    
        recipe = cons()
        # Parse the rest of the options
        (options, noargs) = recipe.parser.parse_args(args=args[1:])
        recipe.run()
        


if __name__ == '__main__':
  main()
