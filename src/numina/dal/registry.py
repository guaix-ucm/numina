#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Persistent registry of the reductions

The registry records the observing blocks, the tasks, the results and
the products of each reduction, so that later reductions (in another
run of numina, a script or a notebook) can use them.

The registry is stored in a JSON file, as collections of documents.
"""

import abc
import contextlib
import copy
import datetime
import json
import logging
import os
import shutil
import tempfile

from numina.util.fqn import fully_qualified_name
from numina.util.jsonencoder import ExtEncoder

_logger = logging.getLogger(__name__)

#: Value of the key ``format`` of the registry file
REGISTRY_FORMAT = "numina-registry"
#: Version of the registry file
REGISTRY_VERSION = 1

#: Metadata of the products stored in the registry
DB_PRODUCT_KEYS = ["instrument", "observation_date", "uuid", "quality_control"]

#: The minimum length of an abbreviated uuid in a reference
UUID_PREFIX_LENGTH = 8

#: The keys of a reference to the registry, in the requirements
REFERENCE_KEYS = ("product", "result", "result_of", "task")


def is_reference(value):
    """True if `value` is a reference to the registry"""
    return isinstance(value, dict) and any(key in value for key in REFERENCE_KEYS)


def atomic_write(filename, write):
    """Write a file atomically

    The contents are written by ``write(fp)`` in a temporary file in the
    same directory, that replaces `filename` only if the write succeeds.
    """
    dirname = os.path.dirname(os.path.abspath(filename))
    fd, tmpname = tempfile.mkstemp(dir=dirname, prefix=".numina-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fp:
            write(fp)
        # keep the permissions of the existing file
        if os.path.exists(filename):
            shutil.copymode(filename, tmpname)
        os.replace(tmpname, filename)
    except BaseException:
        os.unlink(tmpname)
        raise


class Store(abc.ABC):
    """Storage of the registry, as collections of documents

    Each document is a dictionary with an integer ``id``, unique in its
    collection. The methods return copies of the documents. The changes
    made inside :meth:`transaction` are stored together, or discarded if
    it ends with an error.
    """

    @abc.abstractmethod
    def transaction(self):
        """Context manager that groups changes"""

    @abc.abstractmethod
    def insert(self, collection, doc):
        """Insert a copy of `doc` in `collection`, return its id"""

    @abc.abstractmethod
    def update(self, collection, docid, values):
        """Update the fields of a document with `values`"""

    @abc.abstractmethod
    def get(self, collection, docid):
        """Return a document, KeyError if it does not exist"""

    @abc.abstractmethod
    def find(self, collection, **equal):
        """Return the documents whose fields are equal to `equal`, in the order they were inserted"""


class JSONStore(Store):
    """Collections of documents stored in a JSON file

    Each document is a dictionary with an integer ``id``, unique in its
    collection. The ids are not reused, even if documents are deleted.

    The changes are written to the file when the outermost
    :meth:`transaction` ends without errors. If it ends with an
    error, the changes made inside it are discarded.
    """

    def __init__(self, filename):
        """Load the documents of `filename`, if the file exists."""
        self.filename = filename
        self._data = self._empty()
        self._depth = 0
        self._saved = None
        if os.path.exists(filename):
            self._data = self._load(filename)

    @staticmethod
    def _empty():
        """Return the contents of an empty registry."""
        return {"format": REGISTRY_FORMAT, "version": REGISTRY_VERSION, "collections": {}}

    @staticmethod
    def _load(filename):
        """Load the registry stored in `filename`.

        Raises
        ------
        ValueError
            If the file is not a numina registry or its version is not supported.
        """
        with open(filename) as fd:
            data = json.load(fd)
        if not isinstance(data, dict) or data.get("format") != REGISTRY_FORMAT:
            raise ValueError(f"'{filename}' is not a numina registry")
        version = data.get("version")
        if version != REGISTRY_VERSION:
            raise ValueError(f"unsupported version {version} of numina registry '{filename}'")
        return data

    def _collection(self, name):
        """Return the collection `name`, created empty if it does not exist."""
        return self._data["collections"].setdefault(name, {"next_id": 1, "documents": {}})

    @contextlib.contextmanager
    def transaction(self):
        """Group changes, that are written together to the file"""
        if self._depth == 0:
            self._saved = copy.deepcopy(self._data)
        self._depth += 1
        try:
            yield self
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._data = self._saved
                self._saved = None
            raise
        self._depth -= 1
        if self._depth == 0:
            self._saved = None
            self._write()

    def _write(self):
        """Write the documents to the file, atomically."""

        def write(fp):
            json.dump(self._data, fp, indent=2, cls=ExtEncoder)

        atomic_write(self.filename, write)

    def insert(self, collection, doc):
        """Insert a copy of `doc` in `collection`, return its id"""
        with self.transaction():
            coll = self._collection(collection)
            newid = coll["next_id"]
            coll["next_id"] = newid + 1
            stored = copy.deepcopy(doc)
            stored["id"] = newid
            coll["documents"][str(newid)] = stored
        return newid

    def update(self, collection, docid, values):
        """Update the fields of a document with `values`"""
        with self.transaction():
            doc = self._document(collection, docid)
            doc.update(copy.deepcopy(values))
            doc["id"] = docid

    def get(self, collection, docid):
        """Return a copy of a document"""
        return copy.deepcopy(self._document(collection, docid))

    def _document(self, collection, docid):
        """Return a document, not a copy, ``KeyError`` if it does not exist."""
        try:
            return self._collection(collection)["documents"][str(docid)]
        except KeyError:
            raise KeyError(f"document id={docid} not found in collection '{collection}'") from None

    def find(self, collection, **equal):
        """Return copies of the documents whose fields are equal to the values in `equal`

        The documents are returned in the order they were inserted.
        """
        docs = self._collection(collection)["documents"].values()
        result = [doc for doc in docs if all(doc.get(key) == value for key, value in equal.items())]
        result.sort(key=lambda doc: doc["id"])
        return copy.deepcopy(result)


def _format_time(value):
    """Time in ISO format, None if not set"""
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if not value:
        return None
    return str(value)


class Registry:
    """Registry of the reductions made by numina

    The collections of the registry are:

    oblocks
        Observing blocks, with the definition used in the last reduction.
    tasks
        Reduction tasks, with a copy of the observing block they processed.
    results
        Results of the tasks that finished.
    products
        Products of the results, that can be used by other reductions.
    """

    def __init__(self, filename, basedir=None, store=None):
        """Open the registry stored in `filename`.

        `basedir` is the directory the paths of the results are relative to,
        the current directory by default. `store` replaces the default
        ``JSONStore`` of `filename`.
        """
        self.filename = filename
        #: Base directory, the paths of the results are relative to it
        self.basedir = os.getcwd() if basedir is None else basedir
        #: Storage of the documents, a :class:`Store` (JSONStore by default)
        self.store = JSONStore(filename) if store is None else store

    def new_task(self, task, oblock):
        """Record a new task, return its id

        The observing block is recorded as the current definition
        of the OB, and the task keeps a copy of it.
        """
        oblock_id = task.request_params["oblock_id"]
        with self.store.transaction():
            self._record_oblock(oblock_id, oblock)
            doc = {
                "oblock_id": oblock_id,
                "oblock": oblock,
                "state": task.state,
                "time_create": _format_time(task.time_create),
                "time_start": None,
                "time_end": None,
                "request": task.request,
                "request_params": task.request_params,
                "request_runinfo": task.request_runinfo,
            }
            return self.store.insert("tasks", doc)

    def record_oblock(self, oblock):
        """Record the definition of an observing block

        Returns "new", "updated" (the definition has changed) or "unchanged".
        """
        with self.store.transaction():
            return self._record_oblock(oblock["id"], oblock)

    def _record_oblock(self, oblock_id, oblock):
        """Record the definition of an observing block, return ``new``, ``updated`` or ``unchanged``."""
        existing = self.store.find("oblocks", oblock_id=oblock_id)
        doc = {"oblock_id": oblock_id, "definition": oblock}
        if not existing:
            self.store.insert("oblocks", doc)
            return "new"
        if existing[0]["definition"] != oblock:
            _logger.info("oblock_id=%s has changed, updating the registry", oblock_id)
            self.store.update("oblocks", existing[0]["id"], doc)
            return "updated"
        return "unchanged"

    def oblock(self, oblock_id):
        """The definition of the observing block `oblock_id`, None if it is not recorded"""
        existing = self.store.find("oblocks", oblock_id=oblock_id)
        if not existing:
            return None
        return copy.deepcopy(existing[0]["definition"])

    def update_task(self, task):
        """Record the state of a task"""
        values = {
            "state": task.state,
            "time_start": _format_time(task.time_start),
            "time_end": _format_time(task.time_end),
            "request_params": task.request_params,
            "request_runinfo": task.request_runinfo,
        }
        self.store.update("tasks", task.id, values)

    def new_result(self, task, serialized, filename):
        """Record the result of a task and its products, return the id of the result

        Parameters
        ----------
        task : ProcessingTask
            A finished task, with a result
        serialized : dict
            The result as stored in `filename`
        filename : str
            Name of the file of the result, in the results directory of the task
        """
        result = task.result
        runinfo = task.request_runinfo
        res_dir = runinfo["results_dir"]
        oblock_id = task.request_params["oblock_id"]
        with self.store.transaction():
            result_doc = {
                "task_id": task.id,
                "oblock_id": oblock_id,
                "uuid": str(result.uuid),
                "qc": result.qc.name,
                "mode": runinfo["mode"],
                "instrument": runinfo["instrument"],
                "profile": task.request_params["instrument_configuration"],
                "pipeline": runinfo["pipeline"],
                "recipe_class": runinfo["recipe_class"],
                "recipe_fqn": runinfo["recipe_fqn"],
                "time_create": _format_time(task.time_end),
                "result_dir": res_dir,
                "result_file": filename,
            }
            result_id = self.store.insert("results", result_doc)

            for key, prod in result.stored().items():
                if not prod.type.isproduct():
                    continue
                value = getattr(result, key)
                meta = prod.type.extract_db_info(value, DB_PRODUCT_KEYS)
                prod_doc = {
                    "origin": "reduction",
                    "priority": 0,
                    "result_id": result_id,
                    "task_id": task.id,
                    "oblock_id": oblock_id,
                    "field": key,
                    "instrument": runinfo["instrument"],
                    "profile": task.request_params["instrument_configuration"],
                    "type": prod.type.name(),
                    "type_fqn": fully_qualified_name(prod.type),
                    "uuid": meta.get("uuid"),
                    "qc": result.qc.name,
                    "tags": meta.get("tags", {}),
                    "time_create": _format_time(task.time_end),
                    "time_obs": _format_time(meta.get("observation_date")),
                    "content": os.path.join(res_dir, serialized["values"][key]),
                }
                self.store.insert("products", prod_doc)
        return result_id

    def select_product(self, instrument, profile, valid=None, date=None):
        """The product with the highest priority, with QC not BAD

        The product has the given instrument and profile, and
        ``valid(product)`` is True, if `valid` is not None.

        Among the products with the same priority, the product observed
        closest to `date` (a datetime), if it is not None, and then the
        most recent.
        """
        candidates = [
            prod
            for prod in self.store.find("products", instrument=instrument, profile=profile)
            if prod["qc"] != "BAD" and (valid is None or valid(prod))
        ]
        if not candidates:
            return None

        def rank(prod):
            distance = _time_distance(prod.get("time_obs"), date)
            return (-prod.get("priority", 0), distance, -prod["id"])

        return min(candidates, key=rank)

    def set_priority(self, product, priority):
        """Set the priority of a product, by id (integer) or uuid (string), return its id

        The automatic selection of products uses the product with the
        highest priority, 0 by default.
        """
        doc = self._find_by_id_or_uuid("products", "product", product)
        with self.store.transaction():
            self.store.update("products", doc["id"], {"priority": int(priority)})
        return doc["id"]

    def select_result(self, oblock_id):
        """The most recent result of an OB, with QC not BAD"""
        candidates = [res for res in self.store.find("results", oblock_id=oblock_id) if res["qc"] != "BAD"]
        return _most_recent(candidates)

    def select_results(self, instrument, profile, mode):
        """Results of a mode, with QC not BAD, the most recent first

        The results have the given instrument and profile.
        """
        candidates = [
            res
            for res in self.store.find("results", instrument=instrument, mode=mode)
            if res["qc"] != "BAD" and self._result_profile(res) == profile
        ]
        return sorted(candidates, key=lambda doc: doc["id"], reverse=True)

    def _result_profile(self, result):
        """Profile of a result, from its task if it was recorded without it"""
        if "profile" in result:
            return result["profile"]
        try:
            task = self.store.get("tasks", result["task_id"])
        except KeyError:
            return None
        return task["request_params"].get("instrument_configuration")

    def resolve_reference(self, reference, field, types=None):
        """The value of a reference to the registry, and the details of its source

        `reference` is a dictionary with one of the keys:

        product
            The id (integer) or the uuid (string) of a product. The uuid
            can be abbreviated, with at least its first 8 characters.
        result
            The id (integer) or the uuid (string) of a result, as for
            products.
        result_of
            The id of an observing block, its most recent result with QC
            not BAD.
        task
            The id of a task, its result.

        For results, ``field`` is the field of the result, by default
        `field`. The values that are files are returned as absolute paths.
        The details include the tags of the product, if the value is a
        product.

        If `types` is not None, the names of the types of products
        accepted: the product, or the product of the field of the result,
        must be of one of them.

        Raises
        ------
        ValueError
            If the reference is not valid, or what it refers to is not in
            the registry.
        """
        kinds = [key for key in REFERENCE_KEYS if key in reference]
        unknown = set(reference) - set(REFERENCE_KEYS) - {"field"}
        if unknown or len(kinds) != 1:
            raise ValueError(f"invalid reference {reference}, use one of {', '.join(REFERENCE_KEYS)}")
        kind = kinds[0]
        key = reference[kind]
        if kind == "product":
            if "field" in reference:
                raise ValueError(f"invalid reference {reference}, a product has no field")
            doc = self._find_by_id_or_uuid("products", "product", key)
            _check_type(f"the product {key}", doc, types)
            if doc["qc"] == "BAD":
                _logger.warning("the product %s has QC BAD", key)
            return self._path(doc["content"]), {"product_id": doc["id"], "tags": doc.get("tags") or {}}

        if kind == "result_of":
            result = self.select_result(key)
            if result is None:
                # the ids can be integers in a file and strings in other
                matching = [res for res in self.store.find("results") if str(res["oblock_id"]) == str(key)]
                result = _most_recent([res for res in matching if res["qc"] != "BAD"])
            if result is None:
                raise ValueError(f"no result of oblock_id={key} with QC not BAD in the registry")
        elif kind == "result":
            result = self._find_by_id_or_uuid("results", "result", key)
        else:
            result = self._result_of_task(key)
        if kind != "result_of" and result["qc"] == "BAD":
            _logger.warning("the result %s has QC BAD", result["id"])
        field = reference.get("field", field)
        value = self._result_value(result, field)
        # the type is known if the field is a product
        products = self.store.find("products", result_id=result["id"], field=field)
        details = {"result_id": result["id"]}
        if products:
            _check_type(f"the field {field} of the result {result['id']}", products[0], types)
            details["tags"] = products[0].get("tags") or {}
        return value, details

    def _find_by_id_or_uuid(self, collection, what, key):
        """A document by id (integer) or uuid (string), that can be abbreviated"""
        if isinstance(key, int):
            docs = self.store.find(collection, id=key)
            if not docs:
                msg = f"no {what} with id {key} in the registry"
                if len(str(key)) >= UUID_PREFIX_LENGTH:
                    # YAML reads a uuid of digits as an integer
                    msg += f", for a {what} with uuid {key} use {{{what}: '{key}'}}, with quotes"
                raise ValueError(msg)
            return docs[0]
        key = str(key)
        docs = self.store.find(collection, uuid=key)
        if docs:
            return docs[0]
        if len(key) < UUID_PREFIX_LENGTH:
            msg = f"no {what} with uuid {key} in the registry"
            if key.isdigit():
                msg += f", for the {what} with id {key} use {{{what}: {key}}}, without quotes"
            else:
                msg += f", an abbreviated uuid needs at least {UUID_PREFIX_LENGTH} characters"
            raise ValueError(msg)
        docs = [doc for doc in self.store.find(collection) if (doc.get("uuid") or "").startswith(key)]
        if not docs:
            raise ValueError(f"no {what} with uuid {key} in the registry")
        if len(docs) > 1:
            ids = ", ".join(str(doc["id"]) for doc in docs)
            raise ValueError(f"the uuid {key} is ambiguous, it is the beginning of the uuid of the {what}s {ids}")
        return docs[0]

    def _result_of_task(self, task_id):
        """The result of a task"""
        tasks = self.store.find("tasks", id=task_id)
        if not tasks:
            raise ValueError(f"task {task_id} is not in the registry")
        results = self.store.find("results", task_id=task_id)
        if results:
            return results[0]
        if tasks[0]["state"] == 3:
            raise ValueError(f"task {task_id} failed, it has no result")
        raise ValueError(f"task {task_id} has not finished, it has no result")

    def _path(self, name):
        """Return the absolute path of `name`, relative to the base directory."""
        return os.path.abspath(os.path.join(self.basedir, name))

    def _result_value(self, result, field):
        """The value of a field of a result, the files as absolute paths

        The result file does not mark which strings are files: a string
        is a file if it is the name of a file in the directory of the
        result, otherwise it is a value.
        """
        filename = self._path(os.path.join(result["result_dir"], result["result_file"]))
        with open(filename) as fd:
            values = json.load(fd)["values"]
        if field not in values:
            raise ValueError(f"the result {result['id']} has no field {field}")
        value = values[field]
        if isinstance(value, str):
            path = self._path(os.path.join(result["result_dir"], value))
            if os.path.isfile(path):
                return path
        return value

    def tasks(self, **equal):
        """Tasks in the registry, filtered by equality of fields"""
        return self.store.find("tasks", **equal)

    def results(self, **equal):
        """Results in the registry, filtered by equality of fields"""
        return self.store.find("results", **equal)

    def products(self, **equal):
        """Products in the registry, filtered by equality of fields"""
        return self.store.find("products", **equal)

    def oblocks(self, **equal):
        """Observing blocks in the registry, filtered by equality of fields"""
        return self.store.find("oblocks", **equal)


def _check_type(what, product, types):
    """Check that the type of a product is in `types` (if not None)"""

    def strip(label):
        return label[:-2] if label.endswith("()") else label

    if types is not None and strip(product["type"]) not in {strip(label) for label in types}:
        raise ValueError(f"{what} is a {strip(product['type'])}, not a {' or '.join(sorted(types))}")


def _time_distance(time_obs, date):
    """Seconds between the time of a product (ISO string) and date, inf if unknown"""
    if date is None:
        return 0
    if not time_obs:
        return float("inf")
    try:
        value = datetime.datetime.fromisoformat(time_obs)
    except ValueError:
        return float("inf")
    if (value.tzinfo is None) != (date.tzinfo is None):
        value = value.replace(tzinfo=None)
        date = date.replace(tzinfo=None)
    return abs((value - date).total_seconds())


def _most_recent(docs):
    """The document inserted last, None if there are no documents"""
    if not docs:
        return None
    return max(docs, key=lambda doc: doc["id"])
