#
# Copyright 2008-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Default logging configuration for Numina CLI."""


LOGCONF = {
    'version': 1,
    'formatters': {
        'simple': {'format': '%(levelname)s: %(message)s'},
        'state': {'format': '%(asctime)s - %(message)s'},
        'unadorned': {'format': '%(message)s'},
        'detailed': {'format': '%(name)s %(levelname)s %(message)s'},
        'extended': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        },
    'handlers': {
        'unadorned_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'unadorned',
            },
        'simple_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'INFO'
            },
        'simple_console_warnings_only': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'WARNING'
            },
        'detailed_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'INFO'
            },
        },
    'loggers': {
        'numina': {
            'handlers': ['simple_console'],
            'level': 'DEBUG',
            'propagate': False
            }
#        'customdrp': {
#            'handlers': ['detailed_console'],
#            'level': 'DEBUG',
#            'propagate': False
#            }
        },
    'root': {
        'handlers': ['simple_console_warnings_only']
        }
    }
