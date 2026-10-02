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


class JSONStore:
    """Collections of documents stored in a JSON file

    Each document is a dictionary with an integer ``id``, unique in its
    collection. The ids are not reused, even if documents are deleted.

    The changes are written to the file when the outermost
    :meth:`transaction` ends without errors. If it ends with an
    error, the changes made inside it are discarded.
    """

    def __init__(self, filename):
        self.filename = filename
        self._data = self._empty()
        self._depth = 0
        self._saved = None
        if os.path.exists(filename):
            self._data = self._load(filename)

    @staticmethod
    def _empty():
        return {"format": REGISTRY_FORMAT, "version": REGISTRY_VERSION, "collections": {}}

    @staticmethod
    def _load(filename):
        with open(filename) as fd:
            data = json.load(fd)
        if not isinstance(data, dict) or data.get("format") != REGISTRY_FORMAT:
            raise ValueError(f"'{filename}' is not a numina registry")
        version = data.get("version")
        if version != REGISTRY_VERSION:
            raise ValueError(f"unsupported version {version} of numina registry '{filename}'")
        return data

    def _collection(self, name):
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

    def __init__(self, filename, basedir=None):
        self.filename = filename
        #: Base directory, the paths of the results are relative to it
        self.basedir = os.getcwd() if basedir is None else basedir
        self.store = JSONStore(filename)

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

    def _record_oblock(self, oblock_id, oblock):
        existing = self.store.find("oblocks", oblock_id=oblock_id)
        doc = {"oblock_id": oblock_id, "definition": oblock}
        if not existing:
            self.store.insert("oblocks", doc)
        elif existing[0]["definition"] != oblock:
            _logger.info("oblock_id=%s has changed, updating the registry", oblock_id)
            self.store.update("oblocks", existing[0]["id"], doc)

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
                    "result_id": result_id,
                    "task_id": task.id,
                    "oblock_id": oblock_id,
                    "field": key,
                    "instrument": runinfo["instrument"],
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
