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

# $Id: rime-s.py 386 2009-01-20 18:10:57Z spr $

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

def usage():
  print 'uso'

import getopt
import sys

def main():
  try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
  except getopt.GetoptError, err:
    print str(err)
    usage()
    sys.exit(2)
  output = None
  verbose = False
  for o, a in opts:
    if o == "-v":
      verbose = True
    elif o in ("-h", "--help"):
      usage()
      sys.exit()
    elif o in ('-o', '--output'):
      output = a
    else:
      assert False, "unhandled option"

  if len(args) == 0:
    usage()
    sys.exit()
    
  logger.info('Emir recipe runner')
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
  recipe.run()


if __name__ == '__main__':
  main()
