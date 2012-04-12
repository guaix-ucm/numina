#
# Copyright 2011-2012 Universidad Complutense de Madrid
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

'''Numina directories to hold pipelines.'''

import os.path
import sys

from numina.xdgdirs import xdg_data_dirs

__all__ = ['pipeline_path']

# FIXME: we need here install prefix, not the sys.prefix
# still don't now the best way to do it
_path = [os.path.join(sys.prefix, 'share/numina/pipelines')]

_path2 = [os.path.join(base, 'numina/pipelines') for base in xdg_data_dirs]

_path.extend(_path2)

def pipeline_path():
    '''Return a list of directories where look for pipelines.'''
    return _path

if __name__ == '__main__':
    print pipeline_path()
