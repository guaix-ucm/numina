#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DAL for dictionary-based database of products."""

import os
import logging
import json

import yaml

import numina.store
import numina.store.gtc.load as gtcload
from numina.core.oresult import oblock_from_dict
from numina.exceptions import NoResultFound
from numina.instrument.assembly import assembly_instrument, configure_obresult
from numina.util.context import working_directory

from .absdal import AbsDrpDAL
from .stored import StoredProduct, StoredParameter, StoredResult
from .diskfiledal import build_product_path
from .utils import check_template, fill_template, tags_are_valid

_logger = logging.getLogger(__name__)


class BaseDictDAL(AbsDrpDAL):
    """A dictionary based DAL"""

    _RESERVED_MODE_NAMES = ["nulo", "container", "root", "raiz"]

    def __init__(self, drps, ob_table, prod_table, req_table, extra_data=None, components=None):
        """Create the DAL from the tables of observing blocks, products and requirements.

        Parameters
        ----------
        drps : DrpBase
            The DRPs, queried by instrument name.
        ob_table : dict
            Observing blocks, by id.
        prod_table : dict
            Products, by instrument and instrument profile.
        req_table : dict
            Requirements, by instrument, instrument profile, pipeline and mode.
        extra_data : dict, optional
            Values of the requirements given in the command line, by name.
        components : dict, optional
            Components of the instrument configurations.
        """
        super().__init__(drps)
        # Check that the structure of the base is correct
        self.ob_table = ob_table
        self.prod_table = prod_table
        self.req_table = req_table
        self.extra_data = extra_data if extra_data else {}
        self.components = components if components else {}
        #: Where the values returned by the searches were found, by name
        self.sources = {}

    def record_source(self, name, source, **details):
        """Record where the value of `name` was found"""
        self.__dict__.setdefault("sources", {})[name] = dict(source=source, **details)

    def search_prod_type_tags(self, tipo, ins, profile, tags, pipeline):
        """Returns the first coincidence..."""

        drp = self.drps.query_by_name(ins)
        label = drp.product_label(tipo)

        # Strip () is present
        if label.endswith("()"):
            label_alt = label[:-2]
        else:
            label_alt = label

        # search results of these OBs
        ins_table = self.prod_table.get(ins, {})
        ptable = ins_table.get(profile, [])
        for prod in ptable:
            pk = prod["type"]
            pt = prod["tags"]
            if ((pk == label) or (pk == label_alt)) and tags_are_valid(pt, tags):
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)
                rprod["content"] = numina.store.load(tipo, prod["content"])
                return StoredProduct(**rprod)
        else:
            msg = f"type {tipo} compatible with tags {tags!r} not found"
            raise NoResultFound(msg)

    def has_extra(self, name):
        """Check whether a value for requirement `name` was given in the command line."""
        return name in self.extra_data

    def oblock_from_id(self, obsid):
        """Return the observing block with id `obsid`, built from the table of observing blocks."""

        este = self.ob_table[obsid]
        oblock = oblock_from_dict(este)

        return oblock

    def obsres_from_oblock_id(self, obsid, as_mode=None, profile=None):
        """Build the observation result of an observing block

        If profile is not None, it is the uuid of the instrument configuration,
        otherwise the configuration is selected from the images.
        """

        oblock = self.oblock_from_id(obsid)

        return self.obsres_from_oblock(oblock, as_mode, profile=profile)

    def obsres_from_oblock(self, oblock, as_mode=None, profile=None):
        """Build the observation result of the observing block `oblock`.

        The observing block is copied, not modified. `as_mode` replaces the
        observing mode and `profile` is the uuid of the requested instrument
        configuration, as in ``obsres_from_oblock_id``. Unless the mode is a
        reserved name, the observation result is passed through the
        ``build_ob`` method of the observing mode.

        Raises
        ------
        ValueError
            If there is no DRP for the instrument.
        """

        from numina.core.oresult import ObservationResult

        # Copy, the oblock is not modified
        # results, requirements and labels are modified later
        obsres = ObservationResult.__new__(ObservationResult)
        obsres.__dict__ = dict(oblock.__dict__)
        obsres.results = dict(oblock.results)
        obsres.requirements = dict(oblock.requirements)
        obsres.labels = dict(oblock.labels)

        obsres.mode = as_mode or obsres.mode
        _logger.debug("obsres_from_oblock id='%s', mode='%s' START", obsres.id, obsres.mode)

        try:
            this_drp = self.drps.query_by_name(obsres.instrument)
        except KeyError:
            raise ValueError(f"no DRP for instrument {obsres.instrument}")

        # Reserved names
        if obsres.mode in self._RESERVED_MODE_NAMES:
            selected_mode = None  # null mode
        else:
            selected_mode = this_drp.modes[obsres.mode]

        if selected_mode:
            # This is used if we pass an option here or
            # if mode.build_ob_options is defined in the mode class
            # seems useless
            obsres = selected_mode.build_ob(obsres, self)

        obsres.requested_profile = profile
        self.assign_configuration(obsres)
        return obsres

    def assign_configuration(self, obsres):
        """Select and configure the instrument configuration of obsres

        The configuration is the one in obsres.requested_profile, if it is
        not None, or it is selected from the first frame of obsres.
        """
        from numina.core.oresult import ObservingBlock

        this_drp = self.drps.query_by_name(obsres.instrument)
        if obsres.requested_profile is not None:
            # select_profile uses this uuid instead of the images
            obsres.profile = obsres.requested_profile
            _logger.debug("assembly instrument model, profile %s", obsres.profile)
        else:
            # The default value, so that select_profile uses the images
            obsres.profile = ObservingBlock().profile
            _logger.debug("assembly instrument model, auto detection")
        configure_obresult(obsres, this_drp, self.components)

    def update_configuration(self, obsres):
        """Select again the configuration, after adding frames to obsres

        The frames added from the results of other OBs (ResultOf) were not
        available when the configuration was selected in obsres_from_oblock.
        """
        if obsres.requested_profile is None:
            _logger.debug("select again the instrument configuration of OB id=%s", obsres.id)
            self.assign_configuration(obsres)

    def assembly_instrument(self, keyval, date, by_key="name"):
        """Assemble the instrument configuration `keyval` (selected by `by_key`) from the components of the DAL."""
        return assembly_instrument(self.components, keyval, date, by_key=by_key)

    def search_product(self, name, tipo, obsres, options=None):
        """Return the product `name` of type `tipo`, from the command line or from the table of products."""
        # returns StoredProduct
        ins = obsres.instrument
        profile = obsres.profile
        tags = obsres.tags
        pipeline = obsres.pipeline
        if name in self.extra_data:
            val = self.extra_data[name]
            content = numina.store.load(tipo, val)
            self.record_source(name, "command line")
            return StoredProduct(id=0, tags={}, content=content)
        else:
            value = self.search_prod_type_tags(tipo, ins, profile, tags, pipeline)
            self.record_source(name, "control file")
            return value

    def search_parameter(self, name, tipo, obsres, options=None):
        """Return the parameter `name` of type `tipo`, from the command line or from the table of requirements.

        Raises
        ------
        NoResultFound
            If the parameter is not in the requirements of the mode, with tags compatible with those of `obsres`.
        """
        # returns StoredProduct
        instrument = obsres.instrument
        profile = obsres.profile
        mode = obsres.mode
        tags = obsres.tags
        pipeline = obsres.pipeline

        req_table_ins = self.req_table.get(instrument, {})
        req_table_ins_p = req_table_ins.get(profile, {})
        req_table_insi_pipe = req_table_ins_p.get(pipeline, {})

        mode_list = req_table_insi_pipe.get(mode, [])
        if name in self.extra_data:
            value = self.extra_data[name]
            content = StoredParameter(value)
            self.record_source(name, "command line")
            return content
        else:
            for prod in mode_list:
                pn = prod["name"]
                pt = prod["tags"]
                if pn == name and tags_are_valid(pt, tags):
                    # We have found the result, no more checks
                    value = numina.store.load(tipo, prod["content"])
                    content = StoredParameter(value)
                    self.record_source(name, "control file")
                    return content
            else:
                msg = f"name {name} compatible with tags {tags!r} not found"
                raise NoResultFound(msg)

    def search_result_relative(self, name, tipo, obsres, result_desc, options=None):
        """Return an empty list, this DAL does not search results of other observing blocks."""
        # mode field node could go together...
        return []


class Dict2DAL(BaseDictDAL):
    """Dictionary based DAL built from the contents of a control file.

    Parameters
    ----------
    drps : DrpBase
        The DRPs, queried by instrument name.
    obtable : dict
        Observing blocks, by id.
    base : dict
        Contents of the control file, with the keys ``products`` and
        ``requirements`` (or ``parameters``).
    extra_data : dict, optional
        Values of the requirements given in the command line, by name.
    components : dict, optional
        Components of the instrument configurations.
    """

    def __init__(self, drps, obtable, base, extra_data=None, components=None):

        prod_table = base.get("products", {})

        if "parameters" in base:
            req_table = base["parameters"]
        else:
            req_table = base.get("requirements", {})

        super().__init__(drps, obtable, prod_table, req_table, extra_data, components=components)

    def new_task_id(self, request, request_params):
        """Return the id of a new task, the id of the observing block for ``reduce`` requests, otherwise 1."""
        if request == "reduce":
            return request_params.get("oblock_id", 1)
        return 1

    def new_task(self, request, request_params):
        """Create a ``ProcessingTask`` for `request`, with the id given by ``new_task_id``."""

        from numina import __version__
        from numina.user.helpers import ProcessingTask

        newidx = self.new_task_id(request, request_params)
        _logger.debug("create task=%s", newidx)
        task = ProcessingTask()
        task.id = newidx
        task.request = request
        task.request_params = request_params
        task.request_runinfo["runner"] = "numina"
        task.request_runinfo["runner_version"] = __version__
        return task

    def update_task(self, task):
        """Do nothing, the tasks are not stored in this DAL."""
        pass

    def update_result(self, task, serialized, filename):
        """Do nothing, the results are not stored in this DAL."""
        pass

    def dump(self, fp):
        """Write the tables of the DAL to the file object `fp`, in YAML."""
        state = self.dump_data()
        yaml.dump(state, fp)

    def dump_data(self):
        """Return the tables of the DAL as a dictionary, in the format of a control file."""
        state = {}
        state["version"] = 1
        state["products"] = self.prod_table
        state["requirements"] = self.req_table
        state["oblocks"] = self.ob_table
        return state


class BaseHybridDAL(Dict2DAL):
    """Dictionary based DAL that also finds products and results in directories.

    Parameters
    ----------
    drps : DrpBase
        The DRPs, queried by instrument name.
    rootdir : str
        Root directory of the calibration products.
    obtable : dict
        Observing blocks, by id.
    base : dict
        Contents of the control file.
    extra_data : dict, optional
        Values of the requirements given in the command line, by name.
    basedir : str, optional
        Directory the directories of the results are relative to,
        the current directory by default.
    components : dict, optional
        Components of the instrument configurations.
    """

    def __init__(
        self,
        drps,
        rootdir,
        obtable,
        base,
        extra_data=None,
        basedir=None,
        components=None,
    ):

        self.rootdir = rootdir
        self.ob_ids = []
        #: Registry of the reductions (numina.dal.registry.Registry), or None
        self.registry = None

        if basedir is None:
            self.basedir = os.getcwd()
        else:
            self.basedir = basedir

        super().__init__(drps, obtable, base, extra_data=extra_data, components=components)

    def add_obs(self, obtable):
        """Add the observing blocks of the list `obtable` to the table.

        The observing blocks already in the table are ignored, with a warning.
        The ``parent`` of the children of each added block is set.

        Raises
        ------
        ValueError
            If a child of an observing block is not in the table.
        """
        obdict = {}
        for ob in obtable:
            obid = ob["id"]
            if obid not in self.ob_ids:

                self.ob_ids.append(obid)
                obdict[obid] = ob
            else:
                _logger.warning("oblock_id=%s is already in table", obid)
        self.ob_table.update(obdict)

        # Update parents, the children can be stored previously
        for ob in obdict.values():
            for ch in ob.get("children", []):
                try:
                    self.ob_table[ch]["parent"] = ob["id"]
                except KeyError:
                    msg = f"oblock_id={ch}, child of oblock_id={ob['id']}, not found"
                    raise ValueError(msg) from None

    def search_product(self, name, tipo, obsres, options=None):
        """Return the product `name` of type `tipo`, from the command line, the registry or the table of products."""
        if name in self.extra_data:
            val = self.extra_data[name]
            content = numina.store.load(tipo, val)
            self.record_source(name, "command line")
            return StoredProduct(id=0, tags={}, content=content)
        if self.registry is not None:
            try:
                value = self._search_registry_product(tipo, obsres)
                self.record_source(name, "registry", product_id=value.id)
                return value
            except NoResultFound as err:
                _logger.debug("%s", err)
        return self._search_prod_table(name, tipo, obsres)

    def _search_registry_product(self, tipo, obsres):
        """The most recent product in the registry with QC not BAD

        The product must have the same instrument, profile and type,
        and tags compatible with those of obsres.
        """
        drp = self.drps.query_by_name(obsres.instrument)
        label = drp.product_label(tipo)
        labels = {label, label[:-2] if label.endswith("()") else label}
        pred = tipo.query_expr.fill_placeholders(**obsres.tags)

        def valid(prod):
            if prod["type"] not in labels:
                return False
            # pred.eval must return True, non bool values mean that it is incomplete
            if pred.eval(**prod["tags"]) is True:
                return True
            _logger.debug("product id=%s discarded, its tags %s do not match %s", prod["id"], prod["tags"], obsres.tags)
            return False

        date = self._observation_date(obsres, drp)
        prod = self.registry.select_product(obsres.instrument, obsres.profile, valid, date=date)
        if prod is None:
            msg = f"type {label} compatible with tags {obsres.tags} not found in the registry"
            raise NoResultFound(msg)
        _logger.debug("product id=%s from the registry, %s", prod["id"], prod["content"])
        path = os.path.join(self.registry.basedir, prod["content"])
        content = numina.store.load(tipo, path)
        return StoredProduct(id=prod["id"], content=content, tags=prod["tags"])

    def _search_prod_table(self, name, tipo, obsres):
        """Search the product in the table of products, implemented in subclasses."""
        raise NotImplementedError

    @staticmethod
    def _observation_date(obsres, drp):
        """The observation date of the first image of obsres, a datetime, None if unknown"""
        import datetime

        frame = obsres.get_sample_frame()
        if frame is None or not hasattr(frame, "open"):
            return None
        try:
            img = frame.open()
            try:
                value = drp.datamodel.extractor_map["fits"].extract("observation_date", img)
            finally:
                # Close the file only if it was opened here
                if getattr(frame, "frame", None) is None:
                    img.close()
        except Exception as error:
            _logger.debug("observation date of oblock_id=%s not found: %s", obsres.id, error)
            return None
        if isinstance(value, str):
            try:
                value = datetime.datetime.fromisoformat(value)
            except ValueError:
                return None
        return value if isinstance(value, datetime.datetime) else None

    def product_loader(self, tipo, name, path):
        """Load the product `name` of type `tipo` from `path`, a tuple (path, kind).

        If kind is 0, the file is loaded with ``numina.store.load``. Otherwise,
        it is a GTC result file in JSON and its element `name` is returned.
        """
        path, kind = path
        if kind == 0:
            return numina.store.load(tipo, path)
        else:
            # GTC load
            with open(path) as fd:
                data = json.load(fd)
                inter = gtcload.build_result(data)
                elem = inter["elements"]
                return elem[name]

    def search_previous_obsres(self, obsres, node=None):
        """Return the ids of the observing blocks before `obsres`, the nearest first.

        If `node` is ``prev-rel`` and `obsres` has a parent, the blocks are the
        previous children of the parent. Otherwise, they are the blocks added
        to the DAL before `obsres`.
        """

        if node is None:
            node = "prev"

        if node == "prev-rel":
            # Compute nodes relative to parent
            # unless parent is None, then is equal to prev
            parent_id = obsres.parent
            if parent_id is not None:
                cobsres = self.obsres_from_oblock_id(parent_id)
                subset_ids = cobsres.children
                idx = subset_ids.index(obsres.id)
                return reversed(subset_ids[:idx])
            else:
                return self.search_previous_obsid_all(obsres.id)
        else:
            return self.search_previous_obsid_all(obsres.id)

    def search_previous_obsid_all(self, obsid):
        """Return the ids of the observing blocks added before `obsid`, the nearest first."""
        idx = self.ob_ids.index(obsid)
        return reversed(self.ob_ids[:idx])

    def search_result_id(self, node_id, tipo, field, mode=None):
        """Return the field `field` of the result of the observing block `node_id`, implemented in subclasses."""
        raise NotImplementedError

    def search_result_relative(self, name, tipo, obsres, result_desc, options=None):
        """Return the value of a result of other observing blocks, described by `result_desc`.

        The node of `result_desc` selects the observing blocks: with ``children``,
        a list with the values of the children; with ``prev`` and ``prev-rel``,
        the value of the nearest previous block that has it; with ``last``, the
        value in the most recent result of the mode in the registry.

        Raises
        ------
        NoResultFound
            If the value is not found.
        TypeError
            If the node is unknown.
        """

        _logger.debug("search relative result for %s", name)

        # result_type = DataFrameType()
        result_mode = result_desc.mode
        result_field = result_desc.attr
        result_node = result_desc.node

        ignore_fail = result_desc.ignore_fail

        if result_node == "children":
            # Results are multiple
            # one per children
            _logger.debug("search children nodes of %s", obsres.id)
            results = []
            for c in obsres.children:
                try:
                    st = self.search_result_id(c, tipo, result_field)
                    results.append(st)
                except NoResultFound:
                    if not ignore_fail:
                        raise

            return results
        elif result_node == "prev" or result_node == "prev-rel":
            _logger.debug("search previous nodes of %s", obsres.id)

            # obtain previous nodes
            for previd in self.search_previous_obsres(obsres, node=result_node):
                # print('searching in node', previd)
                try:
                    st = self.search_result_id(previd, tipo, result_field, mode=result_mode)
                    return st
                except NoResultFound:
                    pass

            else:
                raise NoResultFound("value not found in any node")
        elif result_node == "last":
            _logger.debug("search last node of %s", result_mode)
            return self.search_result_last(name, tipo, obsres, result_desc)
        else:
            msg = f"unknown node type {result_node}"
            raise TypeError(msg)

    def search_result_last(self, name, tipo, obsres, result_desc):
        """The most recent result of the mode in the registry that has the field"""
        if self.registry is None:
            msg = (
                f"result of '{result_desc.field}' not found, "
                "node 'last' requires a registry of reductions (numina run --db)"
            )
            raise NoResultFound(msg)
        candidates = self.registry.select_results(obsres.instrument, obsres.profile, result_desc.mode)
        for res in candidates:
            try:
                return self._load_result_field(
                    res["oblock_id"], res["result_dir"], res["result_file"], result_desc.attr
                )
            except NoResultFound:
                # this result does not have the field
                continue
        raise NoResultFound(f"result of '{result_desc.field}' not found in the registry")

    def _load_result_field(self, node_id, directory, filename, field):
        """Load field of the result stored in directory/filename"""
        # The observing block may not be reduced, the search continues
        resultdir = os.path.join(self.basedir, directory)
        if not os.path.isdir(resultdir):
            raise NoResultFound(f"directory {directory} of the result of oblock_id={node_id} not found")
        # change directory to open result file
        with working_directory(resultdir):
            if os.path.exists(filename):
                with open(filename) as fd:
                    result_data = json.load(fd)
            else:
                raise NoResultFound(f"{filename} not found in {directory}")

            stored_result = StoredResult.load_data(result_data)

            try:
                content = getattr(stored_result, field)
            except AttributeError:
                raise NoResultFound(f"no field {field} found in result")

            return StoredProduct(id=node_id, content=content, tags={})

    def build_product_path(self, drp, conf, name, tipo, obsres):
        """Return the path of the product, a tuple (path, kind), searched in the root directory."""
        path = build_product_path(drp, self.rootdir, conf, name, tipo, obsres)
        return path

    def dump_data(self):
        """Return the tables of the DAL, with the root directory and the ids of the observing blocks."""
        state = super().dump_data()
        state["rootdir"] = self.rootdir
        state["ob_ids"] = self.ob_ids
        return state


class HybridDAL(BaseHybridDAL):
    """A DAL that can read files from directory structure"""

    def __init__(
        self,
        drps,
        rootdir,
        obtable,
        base,
        extra_data=None,
        components=None,
        basedir=None,
        resultdir_tmpl="obsid{obsid}_results",
        resultfile_tmpl="result.json",
    ):
        """Create the DAL and add the observing blocks of `obtable`.

        Parameters
        ----------
        drps, rootdir, base, extra_data, components, basedir
            As in ``BaseHybridDAL``.
        obtable : list of dict
            Observing blocks.
        resultdir_tmpl : str, optional
            Template of the directories of the results of the observing blocks,
            with the fields ``{obsid}`` and ``{taskid}``.
        resultfile_tmpl : str, optional
            Template of the names of the result files, with the same fields.
        """

        # Templates to find the results of other OBs,
        # the same used by DataManager to store them
        self.resultdir_tmpl = check_template(resultdir_tmpl)
        self.resultfile_tmpl = check_template(resultfile_tmpl)

        super().__init__(
            drps,
            rootdir,
            {},
            base,
            extra_data=extra_data,
            basedir=basedir,
            components=components,
        )
        self.add_obs(obtable)

    def _search_prod_table(self, name, tipo, obsres):
        """Returns the first coincidence..."""

        instrument = obsres.instrument

        profile = obsres.profile

        drp = self.drps.query_by_name(instrument)
        label = drp.product_label(tipo)
        # Strip () is present
        if label.endswith("()"):
            label_alt = label[:-2]
        else:
            label_alt = label

        # search results of these OBs
        ins_table = self.prod_table.get(instrument, {})
        ptable = ins_table.get(profile, [])
        for prod in ptable:
            pk = prod["type"]
            pt = prod["tags"]
            if ((pk == label) or (pk == label_alt)) and tags_are_valid(pt, obsres.tags):
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)

                if "content" in prod:
                    path = prod["content"]
                    _logger.debug("path is %s", path)
                    rprod["content"] = numina.store.load(tipo, path)
                else:
                    # Build path, a tuple (path, kind)
                    path = self.build_product_path(drp, profile, name, tipo, obsres)
                    _logger.debug("path is %s", path)
                    rprod["content"] = self.product_loader(tipo, name, path)
                self.record_source(name, "control file")
                return StoredProduct(**rprod)
        else:
            # Not in table, try file directly
            _logger.debug("%s not in table, try file directly", tipo)
            path = self.build_product_path(drp, profile, name, tipo, obsres)
            _logger.debug("path is %s", path)
            content = self.product_loader(tipo, name, path)
            self.record_source(name, "calibrations directory", path=path)
            return StoredProduct(id=0, content=content, tags=obsres.tags)

    def search_result_id(self, node_id, tipo, field, mode=None):
        """Return the field `field` of the result of the observing block `node_id`.

        The result is the one recorded in the registry, if there is one.
        Otherwise, it is read from the directory and the file built with the
        templates, with the id of the observing block as id of the task.

        Raises
        ------
        NoResultFound
            If the mode of the observing block is not `mode` (when given),
            or if the result or the field is not found.
        """
        cobsres = self.obsres_from_oblock_id(node_id)

        if mode is not None:
            # mode must match
            if cobsres.mode != mode:
                msg = f"requested mode '{mode}' and obsmode '{cobsres.mode}' do not match"
                raise NoResultFound(msg)

        if self.registry is not None:
            result = self.registry.select_result(node_id)
            if result is not None:
                return self._load_result_field(node_id, result["result_dir"], result["result_file"], field)
            if "{taskid}" in self.resultdir_tmpl:
                # the directory can not be built without the id of the task
                raise NoResultFound(f"no result of oblock_id={node_id} in the registry")

        # Without registry, the id of the task is the id of the OB
        directory = fill_template(self.resultdir_tmpl, obsid=node_id, taskid=node_id)
        filename = fill_template(self.resultfile_tmpl, obsid=node_id, taskid=node_id)
        try:
            return self._load_result_field(node_id, directory, filename, field)
        except KeyError as err:
            msg = f"field '{field}' not found in result of mode '{cobsres.mode}' id={node_id}"
            raise NoResultFound(msg) from err
