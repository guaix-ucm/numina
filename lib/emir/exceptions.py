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

'''Exceptions for the emir package'''

__version__ = "$Revision$"

class Error(Exception):
    """Base class for exceptions in the emir package."""
    def __init__(self, txt):
        Exception.__init__(self, txt)
        
        
class DetectorElapseError(Error):
    def __init__(self, txt):
        Error.__init__(self, txt)


class DetectorReadoutError(Error):
    def __init__(self, txt):
        Error.__init__(self, txt)
