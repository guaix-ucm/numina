#
# Copyright 2017-2019 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import inspect
import numina.exceptions as nexcep
import numina.types.datatype as dt


class MultiType(dt.DataType):
    def __init__(self, *args):

        node_type = []
        for obj in args:
            if inspect.isclass(obj):
                obj = obj()
            node_type.append(obj)

        super(MultiType, self).__init__(ptype=None, node_type=node_type)

    def validate(self, obj):
        if self.internal_type:
            return self.internal_type.validate(obj)
        else:
            faillures = []
            for subtype in self.node_type:
                try:
                    return subtype.validate(obj)
                except nexcep.ValidationError as val_err:
                    faillures.append(val_err)
            else:
                internal_m = tuple(v.args[1] for v in faillures)
                raise nexcep.ValidationError(obj, internal_m)

    def isproduct(self):
        if self.internal_type:
            return self.internal_type.isproduct()
        return True

    def tag_names(self):
        if self.internal_type:
            return self.internal_type.tag_names()
        else:
            join = []
            for subtype in self.node_type:
                join.extend(subtype.tag_names())
            return join

    def descriptive_name(self):
        start, remain = self.node_type[0], self.node_type[1:]
        build_str = [start.descriptive_name()]
        for x in remain:
            field = "or {}".format(x.descriptive_name())
            build_str.append(field)
        return " ".join(build_str)

    def _datatype_load(self, obj):
        faillures = []
        for subtype in self.node_type:
            try:
                return subtype._datatype_load(obj)
            except KeyError:
                faillures.append(subtype)
        else:
            msg = "types {} cannot load 'obj'".format(faillures)
            raise TypeError(msg)

    def _query_on_dal(self, name, dal, ob, options=None):

        # Results for subtypes
        results = []
        faillures = []
        for subtype in self.internal_type:
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
            self.internal_type = subtype
            return result
        else:
            raise nexcep.NoResultFound
