#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

from __future__ import print_function

import datetime
import logging
import os
import errno
import shutil
import pickle

import six
import yaml

from numina.util.jsonencoder import ExtEncoder
import numina.util.objimport as objimp
from numina.types.frame import DataFrameType
from numina.util.context import working_directory


_logger = logging.getLogger(__name__)


class DataManager(object):
    def __init__(self, basedir, datadir, backend):
        self.basedir = basedir
        self.datadir = datadir
        self.backend = backend

        self.workdir_tmpl = "obsid{obsid}_{taskid}_work"
        self.resultdir_tmpl = "obsid{obsid}_{taskid}_result"
        self.serial_format = 'json'

        self.result_file = 'result.json'
        self.task_file = 'task.json'

        self.storage = DiskStorageBase()

    def serializer(self, data, fd):
        if self.serial_format == 'yaml':
            self.serializer_json(data, fd)
        elif self.serial_format == 'json':
            self.serializer_json(data, fd)
        else:
            raise ValueError('serializer not supported')

    def serializer_json(self, data, fd):
        import json
        json.dump(data, fd, indent=2, cls=ExtEncoder)

    def serializer_yaml(self, data, fd):
        import yaml
        yaml.dump(data, fd)

    def store_result_to(self, result, storage):
        import numina.store

        saveres = {}
        saveres['values'] = {}
        # FIXME: workaround for QC, this should be managed elsewhere
        if hasattr(result, 'qc'):
            saveres['qc'] = result.qc

        saveres_v = saveres['values']
        for key, prod in result.stored().items():
            # FIXME: workaround for QC, this should be managed elsewhere
            if key == 'qc':
                continue
            val = getattr(result, key)
            storage.destination = "{}".format(prod.dest)
            saveres_v[key] = numina.store.dump(prod.type, val, storage)

        return saveres

    def store_task(self, task):

        result_dir = task.request_runinfo['results_dir']

        with working_directory(result_dir):
            _logger.info('storing task and result')

            # save to disk the RecipeResult part and return the file to save it
            result_repr = self.store_result_to(task.result, self.storage)

            task_repr = task.__dict__.copy()
            # Change result structure by filename
            task_repr['result'] = self.result_file

            with open(self.result_file, 'w+') as fd:
                self.serializer(result_repr, fd)

            with open(self.storage.task, 'w+') as fd:
                self.serializer(task_repr, fd)

        self.backend.update_task(task)
        self.backend.update_result(task, result_repr)

    def create_workenv(self, task):

        values = dict(
            obsid=task.request_params['oblock_id'],
            taskid=task.id
        )

        work_dir = self.workdir_tmpl.format(**values)
        result_dir = self.resultdir_tmpl.format(**values)

        workenv = BaseWorkEnvironment(
            self.datadir,
            self.basedir,
            work_dir,
            result_dir
        )

        return workenv


class ProcessingTask(object):
    def __init__(self):

        self.result = None
        self.id = 1

        self.time_create = datetime.datetime.utcnow()
        self.time_start = 0
        self.time_end = 0
        self.request = "reduce"
        self.request_params = {}
        self.request_runinfo = {}
        self.state = 0

    def store(self, where):

        # save to disk the RecipeResult part and return the file to save it
        result_repr = self.result.store_to(where)

        import json
        with open(where.result, 'w+') as fd:
            json.dump(result_repr, fd, indent=2, cls=ExtEncoder)

        task_repr = self.__dict__.copy()
        # Change result structure by filename
        task_repr['result'] = where.result

        with open(where.task, 'w+') as fd:
            yaml.dump(task_repr, fd)

        return where.task


class BaseWorkEnvironment(object):
    def __init__(self, datadir, basedir, workdir,
                 resultsdir):

        self.basedir = basedir

        self.workdir_rel = workdir
        self.workdir = os.path.abspath(workdir)

        self.resultsdir_rel = resultsdir
        self.resultsdir = os.path.abspath(resultsdir)

        self.datadir_rel = datadir
        self.datadir = os.path.abspath(datadir)

        if six.PY2:
            index_base = "index-2.pkl"
        else:
            index_base = "index.pkl"

        self.index_file = os.path.join(self.workdir, index_base)
        self.hashes = {}

    def sane_work(self):
        _logger.debug('check workdir for working: %r', self.workdir_rel)
        make_sure_path_exists(self.workdir)
        make_sure_file_exists(self.index_file)
        # Load dictionary of hashes

        with open(self.index_file, 'rb') as fd:
            try:
                self.hashes = pickle.load(fd)
            except EOFError:
                self.hashes = {}
        # make_sure_path_doesnot_exist(self.resultsdir)
        make_sure_file_exists(self.index_file)

        # make_sure_path_doesnot_exist(self.resultsdir)
        _logger.debug('check resultsdir to store results %r', self.resultsdir_rel)
        make_sure_path_exists(self.resultsdir)

    def copyfiles(self, obsres, reqs):

        _logger.info('copying files from %r to %r', self.datadir_rel, self.workdir_rel)

        if obsres:
            self.copyfiles_stage1(obsres)

        self.copyfiles_stage2(reqs)

    def copyfiles_stage1(self, obsres):
        import astropy.io.fits as fits
        _logger.debug('copying files from observation result')
        tails = []
        sources = []
        for f in obsres.images:
            if not os.path.isabs(f.filename):
                complete = os.path.abspath(os.path.join(self.datadir, f.filename))
            else:
                complete = f.filename
            head, tail = os.path.split(complete)
            # initial.append(complete)
            tails.append(tail)
            #            heads.append(head)
            sources.append(complete)

        dupes = self.check_duplicates(tails)

        for src, obj in zip(sources, obsres.images):
            head, tail = os.path.split(src)
            if tail in dupes:
                # extract UUID
                hdr = fits.getheader(src)
                img_uuid = hdr['UUID']
                root, ext = os.path.splitext(tail)
                key = "{}_{}{}".format(root, img_uuid, ext)

            else:
                key = tail
            dest = os.path.join(self.workdir, key)
            # Update filename in DataFrame
            obj.filename = dest
            self.copy_if_needed(key, src, dest)

        if obsres.results:
            _logger.warning("not copying files in 'results")
        return obsres

    def check_duplicates(self, tails):
        seen = set()
        dupes = set()
        for tail in tails:
            if tail in seen:
                dupes.add(tail)
            else:
                seen.add(tail)
        return dupes

    def copyfiles_stage2(self, reqs):
        _logger.debug('copying files from requirements')
        for _, req in reqs.stored().items():
            if isinstance(req.type, DataFrameType):
                value = getattr(reqs, req.dest)
                if value is None:
                    continue

                complete = os.path.abspath(
                    os.path.join(self.datadir, value.filename)
                )

                self.copy_if_needed(value.filename, complete, self.workdir)

    def copy_if_needed(self, key, src, dest):

        md5hash = compute_md5sum_file(src)
        _logger.debug('compute hash, %s %s %s', key, md5hash, src)

        # Check hash
        hash_in_file = self.hashes.get(key)
        if hash_in_file is None:
            trigger_save = True
            make_copy = True
        elif hash_in_file == md5hash:
            trigger_save = False
            if os.path.isfile(dest):
                make_copy = False
            else:
                make_copy = True
        else:
            trigger_save = True
            make_copy = True

        self.hashes[key] = md5hash

        if make_copy:
            _logger.debug('copying %r to %r', key, self.workdir)
            shutil.copy(src, dest)
        else:
            _logger.debug('copying %r not needed', key)

        if trigger_save:
            _logger.debug('save hashes')
            with open(self.index_file, 'wb') as fd:
                pickle.dump(self.hashes, fd)

    def adapt_obsres(self, obsres):
        """Adapt obsres after file copy"""

        _logger.debug('adapt observation result for work dir')
        for f in obsres.images:
            # Remove path components
            f.filename = os.path.basename(f.filename)
        return obsres


class WorkEnvironment(BaseWorkEnvironment):
    def __init__(self, obsid, basedir, workdir=None,
                 resultsdir=None, datadir=None):
        if workdir is None:
            workdir = os.path.join(basedir, 'obsid{}_work'.format(obsid))

        if resultsdir is None:
            resultsdir = os.path.join(basedir, 'obsid{}_results'.format(obsid))

        if datadir is None:
            datadir = os.path.join(basedir, 'data')
        super(WorkEnvironment, self).__init__(datadir, basedir, workdir, resultsdir)


def compute_md5sum_file(filename):
    import hashlib
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


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


def make_sure_file_exists(path):
    try:
        with open(path, 'a') as fd:
            pass
    except (OSError, IOError) as exception:
        if exception.errno != errno.EEXIST:
            raise


class DiskStorageBase(object):
    def __init__(self):
        super(DiskStorageBase, self).__init__()
        self.result = 'result.yaml'
        self.task = 'task.yaml'
        self.idx = 1

    def get_next_basename(self, ext):
        fname = 'product_%03d%s' % (self.idx, ext)
        self.idx = self.idx + 1
        return fname

    def store(self, completed_task, resultsdir):
        """Store the values of the completed task."""

        with working_directory(resultsdir):
            _logger.info('storing result')
            return completed_task.store(self)


class DiskStorageDefault(object):
    # TODO: Deprecate
    def __init__(self, resultsdir):
        super(DiskStorageDefault, self).__init__()
        self.result = 'result.yaml'
        self.task = 'task.yaml'
        self.resultsdir = resultsdir
        self.idx = 1

    def get_next_basename(self, ext):
        fname = 'product_%03d%s' % (self.idx, ext)
        self.idx = self.idx + 1
        return fname

    def store(self, completed_task):
        """Store the values of the completed task."""

        with working_directory(self.resultsdir):
            _logger.info('storing result')
            return completed_task.store(self)
