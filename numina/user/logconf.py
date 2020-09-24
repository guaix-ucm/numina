#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Default logging configuration for Numina CLI."""


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
            },
        'simple_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            },
        'simple_console_warnings_only': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'WARNING'
            },
        'detailed_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            },
        },
    'loggers': {
        'numina': {
            'handlers': ['simple_console'],
            'level': 'INFO',
            'propagate': False
            },
        'numina.dal': {
            'handlers': ['detailed_console'],
            'level': 'INFO',
            'propagate': False
            },
        'numina.recipes': {
            'handlers': ['detailed_console'],
            'level': 'INFO',
            'propagate': False
            },
        },
    'root': {
        'handlers': ['detailed_console'],
        'level': 'INFO'
        }
    }
