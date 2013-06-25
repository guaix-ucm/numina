#
# Copyright 2008-2013 Universidad Complutense de Madrid
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
# along with Numina. If not, see <http://www.gnu.org/licenses/>.
# 

from __future__ import print_function
from __future__ import division

import sys
import logging
import os
import errno
import importlib
import traceback

from .simulation import SimulationBase

LOGCONF = {'version': 1,
  'formatters': {'simple': {'format': '%(levelname)s: %(message)s'},
                 'state': {'format': '%(asctime)s - %(message)s'},
                 'unadorned': {'format': '%(message)s'},
                 'detailed': {'format': '%(name)s %(levelname)s %(message)s'},
                 },
  'handlers': {'unadorned_console':
               {'class': 'logging.StreamHandler',
                'formatter': 'unadorned',
                'level': 'DEBUG'
                 },
               'simple_console':
               {'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'DEBUG'
                 },
               'simple_console_warnings_only':
               {'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'WARNING'
                 },
               'detailed_console':
               {'class': 'logging.StreamHandler',
                'formatter': 'detailed',
                'level': 'DEBUG'
                 },
               },
  'loggers': {'task': {'handlers': ['simple_console'], 'level': 'DEBUG', 'propagate': False},
              'emir': {'handlers': ['detailed_console'], 'level': 'DEBUG', 'propagate': False},
              'emirextras': {'handlers': ['detailed_console'], 'level': 'DEBUG', 'propagate': False},
              'numina': {'handlers': ['detailed_console'], 'level': 'DEBUG', 'propagate': False},
              },
  'root': {'handlers': ['detailed_console'], 'level': 'NOTSET'}
}


_logger = logging.getLogger("task")

def _get_masked_mode(mode):
    mask = os.umask(0)
    os.umask(mask)
    return mode & ~mask

def makedirs(name):
    try:
        os.makedirs(name)
    except OSError as e:
        import stat as st
        if not (e.errno == errno.EEXIST and os.path.isdir(name) and 
                st.S_IMODE(os.lstat(name).st_mode) == _get_masked_mode(0o777)):
            raise

def test_task(main, testmodule, workdir, no_create):
    _logger.info('Running tests of the task')  
    prev = os.getcwd()
    _logger.debug('Our cwd is %s', prev)  
    try:
    # Try to create the directory, fail if we can't
        makedirs(workdir)
        # Changing to work dir
        os.chdir(workdir)

        _thistests = importlib.import_module(testmodule)

        count = 0
        skipped = 0

        for Cls in SimulationBase.__subclasses__():  # @UndefinedVariable
            try:
                thissimulate = Cls()
            except TypeError as exception:
                _logger.warning('Broken test: %s', exception)
                traceback.print_exc()
                continue
            skipthis = getattr(thissimulate, 'skip', False)
            if skipthis:
                skipped += 1
                continue
            count += 1
            # Create if not exists
            workdir = getattr(thissimulate, 'workdir', Cls.__name__)
            base = os.getcwd()
            try:
                makedirs(workdir)
                _logger.debug('Created directory %s', workdir)
                os.chdir(workdir)
                if not no_create:
                    _logger.info('Running simulation %r', Cls.__name__)
                    thissimulate.simulate()
                else:
                    _logger.info('Skipping simulation')
        
                _logger.info('Running myself over artificial data')
                _logger.info('with args %s', thissimulate.command_line)
                main(thissimulate.command_line)

                _logger.info('Checking outputs')
                thissimulate.check()
            finally:
                os.chdir(base)
        if count == 0:
            _logger.warning('No tests defined')
        else:
            _logger.info('Tests run %d', count)
        if skipped > 0:
            _logger.info('Tests skipped %d', skipped)

    except OSError as exception:
        _logger.error("%s", exception)
        sys.exit(1)
    finally:
        # Going back
        os.chdir(prev)

