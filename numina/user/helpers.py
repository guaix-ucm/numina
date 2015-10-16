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

'''User command line interface of Numina.'''

from __future__ import print_function

import logging
import os
import errno
import shutil

from numina import __version__
from numina.core import fully_qualified_name
from numina.core.products import DataFrameType



_logger = logging.getLogger("numina")


class ProcessingTask(object):

    def __init__(self, obsres=None, runinfo={}, insconf=None):

        self.observation = {}

        self.runinfo = {}

        if runinfo:
            self.runinfo['pipeline'] = runinfo['pipeline']
            self.runinfo['recipe'] = runinfo['recipeclass'].__name__
            self.runinfo['recipe_full_name'] = fully_qualified_name(runinfo['recipeclass'])
            self.runinfo['runner'] = 'numina'
            self.runinfo['runner_version'] = __version__
            self.runinfo['data_dir'] = runinfo['workenv'].datadir
            self.runinfo['work_dir'] = runinfo['workenv'].workdir
            self.runinfo['results_dir'] = runinfo['workenv'].resultsdir
            self.runinfo['recipe_version'] = runinfo['recipe_version']

        if obsres:
            self.observation['mode'] = obsres.mode
            self.observation['observing_result'] = obsres.id
            self.observation['instrument'] = obsres.instrument
        else:
            self.observation['mode'] = None
            self.observation['observing_result'] = None
            self.observation['instrument'] = None

        if insconf:
            self.observation['instrument_configuration'] = insconf


class WorkEnvironment(object):
    def __init__(self, basedir, workdir=None,
                 resultsdir=None, datadir=None):

        self.basedir = basedir

        if workdir is None:
            workdir = os.path.join(basedir, '_work')

        self.workdir = os.path.abspath(workdir)

        if resultsdir is None:
            resultsdir = os.path.join(basedir, '_results')

        self.resultsdir = os.path.abspath(resultsdir)

        if datadir is None:
            datadir = os.path.join(basedir, '_data')

        self.datadir = os.path.abspath(datadir)

    def sane_work(self):
        make_sure_path_doesnot_exist(self.workdir)
        _logger.debug('check workdir for working: %r', self.workdir)
        make_sure_path_exists(self.workdir)

        make_sure_path_doesnot_exist(self.resultsdir)
        _logger.debug('check resultsdir to store results %r', self.resultsdir)
        make_sure_path_exists(self.resultsdir)

    def copyfiles(self, obsres, reqs):

        _logger.info('copying files from %r to %r', self.datadir, self.workdir)

        if obsres:
            self.copyfiles_stage1(obsres)

        self.copyfiles_stage2(reqs)

    def copyfiles_stage1(self, obsres):
        _logger.debug('copying files from observation result')
        for f in obsres.images:
            _logger.debug('copying %r to %r', f.filename, self.workdir)
            complete = os.path.abspath(os.path.join(self.datadir, f.filename))
            shutil.copy(complete, self.workdir)

    def copyfiles_stage2(self, reqs):
        _logger.debug('copying files from requirements')
        for _, req in reqs.stored().items():
            if isinstance(req.type, DataFrameType):
                value = getattr(reqs, req.dest)
                if value is not None:
                    _logger.debug('copying %r to %r',value.filename,
                                  self.workdir)
                    complete = os.path.abspath(os.path.join(self.datadir,
                                                            value.filename))
                    shutil.copy(complete, self.workdir)


def make_sure_path_doesnot_exist(path):
    try:
        shutil.rmtree(path)
    except (OSError, IOError) as exception:
        if exception.errno != errno.ENOENT:
            raise


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except (OSError, IOError) as exception:
        if exception.errno != errno.EEXIST:
            raise


class DiskStorage(object):
    def __init__(self):
        self.idx = 1

    def get_next_basename(self, ext):
        fname = 'product_%03d%s' % (self.idx, ext)
        self.idx = self.idx + 1
        return fname


from numina.core.pipeline import init_store_backends
from numina.store import dump


class DiskStorageDefault(DiskStorage):
    def __init__(self, resultsdir):
        super(DiskStorageDefault, self).__init__()
        self.result = 'result.yaml'
        self.task = 'task.yaml'
        self.resultsdir = resultsdir
        init_store_backends()


    def store(self, completed_task):
        '''Store the values of the completed task'''

        try:
            csd = os.getcwd()
            _logger.debug('cwd to resultdir: %r', self.resultsdir)
            os.chdir(self.resultsdir)

            _logger.info('storing result')
            completed_task.store(self)
            #dump(completed_task, completed_task, self)

        finally:
            _logger.debug('cwd to original path: %r', csd)
            os.chdir(csd)
