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

'''Decorators for the emir package'''

import time
import logging

__version__ = "$Revision$"

_logger = logging.getLogger("emir")

def print_timing(func):
    '''Print timing decorator'''
    def wrapper(*arg,**keywords):
        t1 = time.time()
        res = func(*arg,**keywords)
        t2 = time.time()
        print '%s took %0.3f s' % (func.func_name, (t2 - t1))
        return res
    return wrapper

def log_timing(func):
    '''Log timing decorator'''
    def wrapper(*arg,**keywords):
        t1 = time.time()
        res = func(*arg,**keywords)
        t2 = time.time()
        _logger.debug('%s took %0.3f s' % (func.func_name, (t2 - t1)))
        return res
    return wrapper
