#
# Copyright 2011 Universidad Complutense de Madrid
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

from xdg.BaseDirectory import xdg_data_dirs

__all__ = ['pipeline_path']

_path = ['/usr/lib64/numina/pipelines']

_path2 = [os.path.join(base, 'numina/pipelines') for base in xdg_data_dirs]

_path.extend(_path2)

def pipeline_path():
    return _path

if __name__ == '__main__':
    print pipeline_path()
