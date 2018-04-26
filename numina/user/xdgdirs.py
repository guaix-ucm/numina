#
# Copyright 2012-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""
Implementation of some of freedesktop.org Base Directories.

The directories are defined here:

http://standards.freedesktop.org/basedir-spec/

We only require xdg_data_dirs and xdg_config_home
"""

import os

_home = os.environ.get('HOME', '/')

xdg_data_home = os.environ.get(
    'XDG_DATA_HOME',
    os.path.join(_home, '.local', 'share')
    )

xdg_data_dirs = [xdg_data_home] + \
    os.environ.get('XDG_DATA_DIRS', '/usr/local/share:/usr/share').split(':')

xdg_config_home = os.environ.get(
    'XDG_CONFIG_HOME',
    os.path.join(_home, '.config')
    )

xdg_cache_home = os.environ.get(
    'XDG_CACHE_HOME',
    os.path.join(_home, '.cache')
    )
