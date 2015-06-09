#
# Copyright 2008-2015 Universidad Complutense de Madrid
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

'''Default logging configuration for Numina CLI.'''

numina_cli_logconf = {
    'version': 1,
    'formatters': {
        'simple': {'format': '%(levelname)s: %(message)s'},
        'state': {'format': '%(asctime)s - %(message)s'},
        'unadorned': {'format': '%(message)s'},
        'detailed': {'format': '%(name)s %(levelname)s %(message)s'},
        },
    'handlers': {
        'unadorned_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'unadorned',
            'level': 'DEBUG'
            },
        'simple_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'DEBUG'
            },
        'simple_console_warnings_only': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'WARNING'
            },
        'detailed_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'DEBUG'
            },
        },
    'loggers': {
        'numina': {
            'handlers': ['simple_console'],
            'level': 'DEBUG',
            'propagate': False
            },
        'numina.recipes': {
            'handlers': ['detailed_console'],
            'level': 'DEBUG',
            'propagate': False
            },
        },
    'root': {
        'handlers': ['detailed_console'],
        'level': 'NOTSET'
        }
    }

