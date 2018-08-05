#
# Copyright 2017-2018 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import inspect
import numina.exceptions as nexcep


class MultiType(object):
    def __init__(self, *args):
        self.type_options = []
        self._current = None
        self.internal_default = None
        for obj in args:
            if inspect.isclass(obj):
                obj = obj()
            self.type_options.append(obj)

    def isproduct(self):
        if self._current:
            return self._current.isproduct()
        return True

    def query(self, name, dal, ob, options=None):

        # Results for subtypes
        results = []
        faillures = []
        for subtype in self.type_options:
            try:
                result = subtype.query(name, dal, ob, options=options)
                results.append((subtype, result))
            except nexcep.NoResultFound as notfound:
                faillures.append((subtype, notfound))
        else:
            # Not found
            for subtype, notfound in faillures:
                subtype.on_query_not_found(notfound)
        # Select wich of the results we choose
        if results:
            # Select the first, for the moment
            subtype, result = results[0]
            self._current = subtype
            return result
        else:
            raise nexcep.NoResultFound

    def on_query_not_found(self, notfound):
        pass

    def convert_in(self, obj):
        if self._current:
            return self._current.convert_in(obj)

        raise ValueError('No query performed, current type is not set')

    def tag_names(self):
        if self._current:
            return self._current.tag_names()
        else:
            join = []
            for subtype in self.type_options:
                join.extend(subtype.tag_names())
            return join

    def validate(self, obj):
        if self._current:
            return self._current.validate(obj)
        else:
            faillures = []
            for subtype in self.type_options:
                try:
                    return subtype.validate(obj)
                except nexcep.ValidationError as val_err:
                    faillures.append(val_err)
            else:
                internal_m = tuple(v.args[1] for v in faillures)
                raise nexcep.ValidationError(obj, internal_m)


    def descriptive_name(self):
        start, remain = self.type_options[0], self.type_options[1:]
        build_str = [start.descriptive_name()]
        for x in remain:
            field = "or {}".format(x.descriptive_name())
            build_str.append(field)
        return " ".join(build_str)

    def _datatype_load(self, obj):
        faillures = []
        for subtype in self.type_options:
            try:
                return subtype._datatype_load(obj)
            except KeyError:
                faillures.append(subtype)
        else:
            msg = "types {} cannot load 'obj'".format(faillures)
            raise TypeError(msg)