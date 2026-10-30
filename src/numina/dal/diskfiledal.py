#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DAL for file-based database of products."""

import os
import itertools
import logging

# import numina.util.objimport as objimp
from numina.store import load
from numina.exceptions import NoResultFound
from numina.dal import StoredProduct

_logger = logging.getLogger(__name__)


def _combinations(seq):
    """Yield the combinations of the elements of `seq`, from the longest to the empty one."""
    maxl = len(seq)
    for i in range(maxl + 1, -1, -1):
        for j in itertools.combinations(seq, i):
            yield j


class FileFinder:
    """Finder of the files of products in a directory."""

    def candidates(self, directory):
        """Return the names of the files in `directory`, sorted."""
        return [filename for filename in sorted(os.listdir(directory))]

    def build_path(self, directory, value):
        """Return the path of the file `value` in `directory`."""
        return os.path.join(directory, value)

    def build_final_path(self, directory, value):
        """Return the path of the file `value` in `directory`, the result of the search."""
        return os.path.join(directory, value)

    def check(self, directory, value):
        """Check whether `value` is a regular file in `directory`, and not hidden."""
        fname = value
        loadpath = self.build_path(directory, fname)
        # print('loadpath=', loadpath)
        _logger.debug("check %s", loadpath)
        if fname.startswith("."):
            _logger.debug("file %s is hidden, ignore", loadpath)
            return False
        if os.path.isfile(loadpath):
            _logger.debug("is regular file %s", loadpath)
            _logger.info("found %s", loadpath)
            # print(loadpath)
            return True
        else:
            _logger.debug("is not regular file %s", loadpath)
            return False


class FileFinderGTC(FileFinder):
    """Finder of the files of products, including GTC result files.

    The candidates are tuples (name, kind), with kind 1 for the GTC
    result file ``result.json`` and 0 for the other files.
    """

    def candidates(self, directory):
        """Return ``("result.json", 1)`` followed by the files in `directory`, with kind 0."""
        base = [("result.json", 1)]
        other = [(m, 0) for m in super().candidates(directory)]
        base.extend(other)
        return base

    def build_path(self, directory, value):
        return super().build_path(directory, value)

    def build_final_path(self, directory, value):
        """Return a tuple with the path of the candidate `value` in `directory` and its kind."""
        return os.path.join(directory, value[0]), value[1]

    def check(self, directory, value):
        """Check whether the candidate `value` is a regular file in `directory`, and not hidden."""
        fname, kind = value
        return super().check(directory, fname)


def _product_label(drp, tipo, default):
    """The directory of the products of type `tipo`

    It is the alias of the product in the DRP, or `default` if
    no mode of the DRP provides it. The products provided by the
    DRP are indexed by the name of their type.
    """
    try:
        return drp.query_provides(tipo.name()).alias
    except ValueError:
        return default


def build_product_path(drp, rootdir, conf, name, tipo, ob, cls=FileFinderGTC):
    """Search the file of the product `name` of type `tipo` in `rootdir`.

    The directories searched are ``rootdir/instrument/conf/label/tag1/tag2...``,
    where label is the alias of the product in the DRP (or the name of the
    type) and the tags are the values of the tags of `ob`, sorted by name.
    All the combinations of tags are tried, from the longest one.

    Returns
    -------
    str or tuple
        The value returned by ``build_final_path`` of the finder `cls`,
        a tuple (path, kind) with ``FileFinderGTC``.

    Raises
    ------
    NoResultFound
        If no file is found.
    """

    _logger.info("search %s of type %s", name, tipo)

    file_finder = cls()

    label = _product_label(drp, tipo, tipo.name())

    # search results of these OBs
    # build path based in combinations of tags
    vals = [ob.tags[k] for k in sorted(ob.tags.keys())]
    for com in _combinations(vals):
        directory = os.path.join(rootdir, ob.instrument, conf, label, *com)
        _logger.debug("try directory %s", directory)
        try:
            for value in file_finder.candidates(directory):
                if file_finder.check(directory, value):
                    return file_finder.build_final_path(directory, value)
        except OSError as msg:
            _logger.debug("%s", msg)
    else:
        msg = f"type {label} compatible with tags {ob.tags!r} not found"
        _logger.info("%s", msg)
        raise NoResultFound(msg)


DAL_USE_OFFLINE_CALIBS = True


class DiskFileDAL:
    """DAL that loads the products from a directory tree.

    Parameters
    ----------
    drp : InstrumentDRP
        The DRP of the instrument.
    rootdir : str
        Root directory of the products.
    """

    def __init__(self, drp, rootdir, *args, **kwargs):
        super().__init__()
        self.drp = drp
        self.rootdir = rootdir
        self.conf = "default"

    def search_product(self, name, tipo, ob):
        """Return the first file found for the product of type `tipo`, a ``StoredProduct``.

        The directories are those of ``build_product_path``, with the tags in
        their order in `ob` and the configuration ``default``.

        Raises
        ------
        NoResultFound
            If no file is found.
        """
        klass = tipo.__class__

        label = _product_label(self.drp, tipo, klass.__name__)

        # search results of these OBs
        # build path based in combinations of tags
        for com in _combinations(ob.tags.values()):
            directory = os.path.join(self.rootdir, ob.instrument, self.conf, label, *com)
            try:
                files_s = [filename for filename in sorted(os.listdir(directory))]
                for fname in files_s:
                    loadpath = os.path.join(directory, fname)
                    if os.path.isfile(loadpath):
                        if DAL_USE_OFFLINE_CALIBS:
                            content = load(tipo, loadpath)
                            # else:
                            # data = json.load(open(loadpath))
                            # result = process_result(data)
                            # key = self._field_to_extract[klass]
                            # content = result[key]
                            return StoredProduct(id=fname, content=content, tags=ob.tags)
            except OSError:
                # the directory does not exist, search in the next one
                pass
                # msg = 'type %s compatible with tags %r not found' % (klass, ob.tags)
                # raise NoResultFound(msg)
        else:
            msg = f"type {klass} compatible with tags {ob.tags!r} not found"
            raise NoResultFound(msg)
