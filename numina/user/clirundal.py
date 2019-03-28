#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import sys
import logging

import yaml

from numina import __version__
import numina.drps
import numina.exceptions
from numina.dal.dictdal import HybridDAL
from numina.dal.backend import Backend
from numina.util.context import working_directory
from numina.util.fqn import fully_qualified_name
import numina.instrument.assembly as asbl

from .baserun import run_recipe
from .helpers import DataManager


DEFAULT_RECIPE_LOGGER = 'numina.recipes'

_logger = logging.getLogger("numina")


def process_format_version_1(loaded_obs, loaded_data, loaded_data_extra=None, profile_path_extra=None):
    sys_drps = numina.drps.get_system_drps()
    com_store = asbl.load_panoply_store(sys_drps, profile_path_extra)
    backend = HybridDAL(
        sys_drps, [], loaded_data,
        extra_data=loaded_data_extra,
        components=com_store
    )
    backend.add_obs(loaded_obs)
    return backend


def process_format_version_2(loaded_obs, loaded_data, loaded_data_extra=None, profile_path_extra=None):
    sys_drps = numina.drps.get_system_drps()
    com_store = asbl.load_panoply_store(sys_drps, profile_path_extra)
    loaded_db = loaded_data['database']
    backend = Backend(
        sys_drps, loaded_db,
        extra_data=loaded_data_extra,
        components=com_store
    )
    backend.rootdir = loaded_data.get('rootdir', '')
    backend.add_obs(loaded_obs)
    return backend


def mode_run_common(args, extra_args, mode):
    # FIXME: implement 'recipe' run mode
    if mode == 'rec':
        print('Mode not implemented yet')
        return 1
    elif mode == 'obs':
        return mode_run_common_obs(args, extra_args)
    else:
        raise ValueError('Not valid run mode {0}'.format(mode))


def mode_run_common_obs(args, extra_args):
    """Observing mode processing mode of numina."""

    # Loading observation result if exists
    loaded_obs = []
    sessions = []
    if args.session:
        for obfile in args.obsresult:
            _logger.info("session file from %r", obfile)

            with open(obfile) as fd:
                sess = yaml.load(fd)
                sessions.append(sess['session'])
    else:
        for obfile in args.obsresult:
            _logger.info("Loading observation results from %r", obfile)

            with open(obfile) as fd:
                sess = []
                for doc in yaml.load_all(fd):
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
    if args.reqs:
        _logger.info('reading control from %s', args.reqs)
        with open(args.reqs, 'r') as fd:
            loaded_data = yaml.load(fd)
    else:
        _logger.info('no control file')
        loaded_data = {}

    if extra_args.extra_control:
        _logger.info('extra control %s', extra_args.extra_control)
        loaded_data_extra = parse_as_yaml(extra_args.extra_control)
    else:
        loaded_data_extra = None

    control_format = loaded_data.get('version', 1)
    _logger.info('control format version %d', control_format)

    if control_format == 1:
        _backend = process_format_version_1(loaded_obs, loaded_data, loaded_data_extra, args.profilepath)
        datamanager = DataManager(args.basedir, args.datadir, _backend)
        datamanager.workdir_tmpl = "obsid{obsid}_work"
        datamanager.resultdir_tmpl = "obsid{obsid}_results"
        datamanager.serial_format = 'yaml'
        datamanager.result_file = 'result.yaml'
        datamanager.task_file = 'task.yaml'

    elif control_format == 2:
        _backend = process_format_version_2(loaded_obs, loaded_data, loaded_data_extra, args.profilepath)
        datamanager = DataManager(args.basedir, args.datadir, _backend)
    else:
        print('Unsupported format', control_format, 'in', args.reqs)
        sys.exit(1)

    # Start processing
    jobs = []
    for session in sessions:
        for job in session:
            if job['enabled']:
                jobs.append(job)

    for job in jobs:
        # Directories with relevant data
        request = 'reduce'
        request_params = {}

        obid = job['id']

        request_params['oblock_id'] = obid
        request_params["pipeline"] = args.pipe_name
        request_params["instrument_configuration"] = args.insconf

        logger_control = dict(
            default=DEFAULT_RECIPE_LOGGER,
            logfile='processing.log',
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            enabled=True
            )
        request_params['logger_control'] = logger_control

        task = datamanager.backend.new_task(request, request_params)
        task.request = request
        task.request_params = request_params

        task.request_runinfo['runner'] = 'numina'
        task.request_runinfo['runner_version'] = __version__

        _logger.info("procesing OB with id={}".format(obid))
        workenv = datamanager.create_workenv(task)

        task.request_runinfo["results_dir"] = workenv.resultsdir_rel
        task.request_runinfo["work_dir"] = workenv.workdir_rel

        # Roll back to cwd after leaving the context
        with working_directory(workenv.datadir):

            obsres = datamanager.backend.obsres_from_oblock_id(obid, configuration=args.insconf)

            _logger.debug("pipeline from CLI is %r", args.pipe_name)
            pipe_name = args.pipe_name
            obsres.pipeline = pipe_name
            recipe = datamanager.backend.search_recipe_from_ob(obsres)
            _logger.debug('recipe class is %s', recipe.__class__)

            # Enable intermediate results by default
            _logger.debug('enable intermediate results')
            recipe.intermediate_results = True

            # Update runinfo
            _logger.debug('update recipe runinfo')
            recipe.runinfo['runner'] = 'numina'
            recipe.runinfo['runner_version'] = __version__
            recipe.runinfo['task_id'] = task.id
            recipe.runinfo['data_dir'] = workenv.datadir
            recipe.runinfo['work_dir'] = workenv.workdir
            recipe.runinfo['results_dir'] = workenv.resultsdir

            _logger.debug('recipe created')

            try:
                rinput = recipe.build_recipe_input(obsres, datamanager.backend)
            except (ValueError, numina.exceptions.ValidationError) as err:
                _logger.error("During recipe input construction")
                _logger.error("%s", err)
                sys.exit(0)
            _logger.debug('recipe input created')

            # Show the actual inputs
            for key in recipe.requirements():
                v = getattr(rinput, key)
                _logger.debug("recipe requires %r, value is %s", key, v)

            for req in recipe.products().values():
                _logger.debug('recipe provides %s, %s', req.type.__class__.__name__, req.description)

        # Load recipe control and recipe parameters from file
        task.request_runinfo['instrument'] =  obsres.instrument
        task.request_runinfo['mode'] = obsres.mode
        task.request_runinfo['recipe_class'] =  recipe.__class__.__name__
        task.request_runinfo['recipe_fqn'] = fully_qualified_name(recipe.__class__)
        task.request_runinfo['recipe_version'] =  recipe.__version__

        # Copy files
        if args.copy_files:
            _logger.debug('copy files to work directory')
            workenv.sane_work()
            workenv.copyfiles_stage1(obsres)
            workenv.copyfiles_stage2(rinput)
            workenv.adapt_obsres(obsres)

        completed_task = run_recipe(recipe=recipe, task=task, rinput=rinput,
                                    workenv=workenv, logger_control=logger_control)

        datamanager.store_task(completed_task)

    if args.dump_control:
        _logger.debug('dump control status')
        with open('control_dump.yaml', 'w') as fp:
            datamanager.backend.dump(fp)


def parse_as_yaml(strdict):
    """Parse a dictionary of strings as if yaml reads it"""
    interm = ""
    for key, val in strdict.items():
        interm = "%s: %s, %s" % (key, val, interm)
    fin = '{%s}' % interm

    return yaml.load(fin)