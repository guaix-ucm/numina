#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
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
    maxl = len(seq)
    for i in range(maxl + 1, -1, -1):
        for j in itertools.combinations(seq, i):
            yield j


class FileFinder(object):

    def candidates(self, directory):
        return [filename for filename in sorted(os.listdir(directory))]

    def build_path(self, directory, value):
        return os.path.join(directory, value)

    def build_final_path(self, directory, value):
        return os.path.join(directory, value)

    def check(self, directory, value):
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

    def candidates(self, directory):
        base = [('result.json', 1)]
        other = [(m, 0) for m in super(FileFinderGTC, self).candidates(directory)]
        base.extend(other)
        return base

    def build_path(self, directory, value):
        return super(FileFinderGTC, self).build_path(directory, value)

    def build_final_path(self, directory, value):
        return os.path.join(directory, value[0]), value[1]

    def check(self, directory, value):
        fname, kind = value
        return super(FileFinderGTC, self).check(directory, fname)


def build_product_path(drp, rootdir, conf, name, tipo, ob, cls=FileFinderGTC):

    _logger.info('search %s of type %s', name, tipo)

    file_finder = cls()

    try:
        # FIXME
        res = drp.query_provides(tipo.__class__)
        label = res.alias
    except ValueError:
        label = tipo.name()

    # search results of these OBs
    # build path based in combinations of tags
    vals = [ob.tags[k] for k in sorted(ob.tags.keys())]
    for com in _combinations(vals):
        directory = os.path.join(rootdir, ob.instrument, conf, label, *com)
        _logger.debug('try directory %s', directory)
        try:
            for value in file_finder.candidates(directory):
                if file_finder.check(directory, value):
                    return file_finder.build_final_path(directory, value)
        except OSError as msg:
            _logger.debug("%s", msg)
    else:
        msg = f'type {label} compatible with tags {ob.tags!r} not found'
        _logger.info("%s", msg)
        raise NoResultFound(msg)


DAL_USE_OFFLINE_CALIBS = True


class DiskFileDAL(object):

    def __init__(self, drp, rootdir, *args, **kwargs):
        super(DiskFileDAL, self).__init__()
        self.drp = drp
        self.rootdir = rootdir
        self.conf = 'default'

    def search_product(self, name, tipo, ob):
        klass = tipo.__class__
        print ('Init search ', name, tipo, tipo.__class__)

        try:
            res = self.drp.query_provides(tipo.__class__)
            label = res.alias
        except ValueError:
            label = tipo.__class__.__name__

        # search results of these OBs
        # build path based in combinations of tags
        for com in _combinations(ob.tags.values()):
            directory = os.path.join(self.rootdir, ob.instrument, self.conf, label, *com)
            print('search in', directory)
            try:
                files_s = [filename for filename in sorted(os.listdir(directory))]
                print("files_s", files_s)
                for fname in files_s:
                    loadpath = os.path.join(directory, fname)
                    print("check ", loadpath)
                    if os.path.isfile(loadpath):
                        print("is regular file ", loadpath)
                        if DAL_USE_OFFLINE_CALIBS:
                            content = load(tipo, loadpath)
                        #else:
                            #data = json.load(open(loadpath))
                            #result = process_result(data)
                            #key = self._field_to_extract[klass]
                            #content = result[key]
                            return StoredProduct(id=files_s[-1], content=content, tags=ob.tags)
                        else:
                            print("not ready")
                    else:
                        print("is not regular file ", loadpath)
            except OSError as msg:
                print(msg)
                #msg = 'type %s compatible with tags %r not found' % (klass, ob.tags)
                #raise NoResultFound(msg)
        else:
            msg = f'type {klass} compatible with tags {ob.tags!r} not found'
            print(msg)
            raise NoResultFound(msg)
