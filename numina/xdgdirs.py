#
# Copyright 2012-2013 Universidad Complutense de Madrid
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

'''
Implementation of some of freedesktop.org Base Directories.

The directories are defined here:

http://standards.freedesktop.org/basedir-spec/

We only require xdg_data_dirs and xdg_config_home
'''

import os

_home = os.environ.get('HOME', '/')

xdg_data_home = os.environ.get('XDG_DATA_HOME',
            os.path.join(_home, '.local', 'share'))

xdg_data_dirs = [xdg_data_home] + \
    os.environ.get('XDG_DATA_DIRS', '/usr/local/share:/usr/share').split(':')

xdg_config_home = os.environ.get('XDG_CONFIG_HOME',
            os.path.join(_home, '.config'))
