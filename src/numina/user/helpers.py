#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import datetime
import errno
import logging
import os
import pickle
import shutil

import yaml

import numina.drps
from numina.dal.backend import Backend
from numina.util.jsonencoder import ExtEncoder
from numina.types.frame import DataFrameType
from numina.types.qc import QC
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

    def insert_obs(self, loaded_obs):
        self.backend.add_obs(loaded_obs)

    def load_observations(self, obfile):

        loaded_obs = []
        with open(obfile) as fd:
            sess = []
            for doc in yaml.load_all(fd):
                enabled = doc.get('enabled', True)
                docid = doc['id']
                requirements = doc.get('requirements', {})
                sess.append(
                    dict(id=docid, enabled=enabled,
                         requirements=requirements)
                )
                loaded_obs.append(doc)
        self.insert_obs(loaded_obs)
        return loaded_obs

    def serializer(self, data, fd):
        if self.serial_format == 'yaml':
            self.serializer_yaml(data, fd)
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

    def store_result_to(self, result):
        saveres = result.store_to(None)
        return saveres

    def store_task(self, task):

        result_dir_rel = task.request_runinfo['results_dir']
        result_dir = os.path.join(self.basedir, result_dir_rel)

        with working_directory(result_dir):

            task_repr = task.__dict__.copy()

            # save to disk the RecipeResult part and return the file to save it
            if task.result is not None:
                uuid = task.result.uuid
                qc = task.result.qc
                _logger.info('storing result uuid=%s, quality=%s', uuid, qc)
                if qc != QC.GOOD:
                    for key, val in task.result.stored().items():
                        val = getattr(task.result, key)
                        if hasattr(val, 'quality_control'):
                            _logger.info('with field %s=%s, quality=%s', key, val, val.quality_control)

                result_repr = self.store_result_to(task.result)
                # Change result structure by filename
                task_repr['result'] = self.result_file

                with open(self.result_file, 'w+') as fd:
                    self.serializer(result_repr, fd)

            tid = task.id
            _logger.info('storing task id=%s', tid)
            with open(self.task_file, 'w+') as fd:
                self.serializer(task_repr, fd)

        self.backend.update_task(task)
        if task.result is not None:
            self.backend.update_result(task, result_repr, self.result_file)

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
        self.request_runinfo = self._init_runinfo()
        self.state = 0

    @classmethod
    def _init_runinfo(cls):
        request_runinfo = dict(runner='unknown', runner_version='unknown')
        return request_runinfo


class BaseWorkEnvironment(object):
    def __init__(self, datadir, basedir, workdir,
                 resultsdir):

        self.basedir = basedir

        self.workdir_rel = workdir
        self.workdir = os.path.abspath(os.path.join(self.basedir, workdir))

        self.resultsdir_rel = resultsdir
        self.resultsdir = os.path.abspath(os.path.join(self.basedir, resultsdir))

        self.datadir_rel = datadir
        self.datadir = os.path.abspath(datadir)

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

    def installfiles_stage1(self, obsres, action='copy'):
        import astropy.io.fits as fits

        install_if_needed = self._calc_install_if_needed(action)

        _logger.debug('installing files from observation result')
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
                key = f"{root}_{img_uuid}{ext}"

            else:
                key = tail
            dest = os.path.join(self.workdir, key)
            # Update filename in DataFrame
            obj.filename = dest
            install_if_needed(key, src, dest)

        if obsres.results:
            _logger.warning("not installing files in 'results")
        return obsres

    def _calc_install_if_needed(self, action):
        if action not in ['copy', 'link']: # , 'symlink', 'hardlink']:
            raise ValueError(f"{action} action is not allowed")

        _logger.debug(f'installing files with "{action}"')

        if action == 'copy':
            install_if_needed = self.copy_if_needed
        elif action == 'link':
            install_if_needed = self.link_if_needed
        else:
            raise ValueError(f"{action} action is not allowed")
        return install_if_needed

    def installfiles_stage2(self, reqs, action='copy'):
        _logger.debug('installing files from requirements')

        install_if_needed = self._calc_install_if_needed(action)

        for _, req in reqs.stored().items():
            if isinstance(req.type, DataFrameType):
                value = getattr(reqs, req.dest)
                if value is None:
                    continue

                complete = os.path.abspath(
                    os.path.join(self.datadir, value.filename)
                )

                head, tail = os.path.split(value.filename)
                dest = os.path.join(self.workdir, tail)

                install_if_needed(value.filename, complete, dest)

    def copyfiles_stage1(self, obsres):
        return self.installfiles_stage1(obsres, action='copy')

    def linkfiles_stage1(self, obsres):
        return self.installfiles_stage1(obsres, action='link')

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
        return self.installfiles_stage2(reqs, action='copy')

    def linkfiles_stage2(self, reqs):
        return self.installfiles_stage2(reqs, action='link')

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

    def link_if_needed(self, key, src, dest):
        _logger.debug('linking %r to %r', key, self.workdir)
        try:
            # Remove destination
            os.remove(dest)
        except OSError:
            pass
        os.symlink(src, dest)

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
            workdir = os.path.join(basedir, f'obsid{obsid}_work')

        if resultsdir is None:
            resultsdir = os.path.join(basedir, f'obsid{obsid}_results')

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


def process_format_version_1(basedir, loaded_data, loaded_data_extra=None, profile_path_extra=None):
    import numina.instrument.assembly as asbl
    from numina.dal.dictdal import HybridDAL
    sys_drps = numina.drps.get_system_drps()
    com_store = asbl.load_panoply_store(sys_drps, profile_path_extra)
    backend = HybridDAL(
        sys_drps, [], loaded_data,
        extra_data=loaded_data_extra,
        basedir=basedir,
        components=com_store
    )
    return backend


def process_format_version_2(basedir, loaded_data, loaded_data_extra=None,
                             profile_path_extra=None, filename=None):
    import numina.instrument.assembly as asbl
    sys_drps = numina.drps.get_system_drps()
    com_store = asbl.load_panoply_store(sys_drps, profile_path_extra)
    loaded_db = loaded_data['database']
    backend = Backend(
        sys_drps, loaded_db,
        extra_data=loaded_data_extra,
        basedir=basedir,
        components=com_store,
        filename=filename
    )
    backend.rootdir = loaded_data.get('rootdir', '')
    return backend


def create_datamanager(reqfile, basedir, datadir,
                       extra_control=None, profile_path_extra=None,
                       persist=True):
    if reqfile:
        _logger.info('reading control from %s', reqfile)
        with open(reqfile, 'r') as fd:
            loaded_data = yaml.safe_load(fd)
    else:
        _logger.info('no control file')
        loaded_data = {}

    if extra_control:
        _logger.info('extra control %s', extra_control)
        loaded_data_extra = parse_as_yaml(extra_control)
    else:
        loaded_data_extra = None

    control_format = loaded_data.get('version', 1)
    _logger.info('control format version %d', control_format)

    if control_format == 1:
        _backend = process_format_version_1(basedir, loaded_data, loaded_data_extra, profile_path_extra)
        datamanager = DataManager(basedir, datadir, _backend)
        datamanager.workdir_tmpl = "obsid{obsid}_work"
        datamanager.resultdir_tmpl = "obsid{obsid}_results"
    elif control_format == 2:
        if persist:
            pname = reqfile
        else:
            pname = None
        _backend = process_format_version_2(basedir, loaded_data,
                                            loaded_data_extra,
                                            profile_path_extra,
                                            filename=pname)


        datamanager = DataManager(basedir, datadir, _backend)
    else:
        print('Unsupported format', control_format, 'in', reqfile)
        raise ValueError
    return datamanager


def load_observations(obfiles, is_session=False):
    """Observing mode processing mode of numina."""

    # Loading observation result if exists
    loaded_obs = []
    sessions = []

    for obfile in obfiles:
        with open(obfile) as fd:
            if is_session:
                _logger.info("session file from %r", obfile)
                sess = yaml.safe_load(fd)
                sessions.append(sess['session'])
            else:
                _logger.info("observation results from %r", obfile)
                sess = []
                for doc in yaml.safe_load_all(fd):
                    enabled = doc.get('enabled', True)
                    docid = doc['id']
                    requirements = doc.get('requirements', {})
                    sess.append(dict(id=docid, enabled=enabled,
                                     requirements=requirements))
                    if enabled:
                        _logger.debug("load observation result with id %s", docid)
                    else:
                        _logger.debug("skip observation result with id %s", docid)

                    loaded_obs.append(doc)

            sessions.append(sess)
    return sessions, loaded_obs


def parse_as_yaml(strdict):
    """Parse a dictionary of strings as if yaml reads it"""
    interm = ""
    for key, val in strdict.items():
        interm = f"{key}: {val}, {interm}"
    fin = '{%s}' % interm

    return yaml.load(fin)
