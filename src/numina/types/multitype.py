#
# Copyright 2017-2019 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import inspect

import numina.exceptions as nexcep
import numina.types.datatype as dt


class MultiType(dt.DataType):
    """Data type whose values can be of any of several types.

    Parameters
    ----------
    *args
        The types, or their classes, which are instantiated.
    """

    def __init__(self, *args):

        node_type = []
        for obj in args:
            if inspect.isclass(obj):
                obj = obj()
            node_type.append(obj)

        super().__init__(ptype=None, node_type=node_type)

    def validate(self, obj):
        """Validate `obj` with the first of the types that accepts it.

        Raises
        ------
        numina.exceptions.ValidationError
            If none of the types accepts `obj`.
        """
        for subtype in self.node_type:
            try:
                return subtype.validate(obj)
            except nexcep.ValidationError:
                pass
        # All the subtypes failed. The arguments are those of
        # DataType.validate, the types are taken from the subtypes,
        # as not every validate raises with the type in args[1]
        internal_m = tuple(subtype.internal_type for subtype in self.node_type)
        raise nexcep.ValidationError(obj, internal_m)

    def isproduct(self):
        """Return True, a MultiType is treated as a data product."""
        return True

    def tag_names(self):
        """Return the tag names of all the types."""
        join = []
        for subtype in self.node_type:
            join.extend(subtype.tag_names())
        return join

    def extract_tags(self, obj):
        """Extract tags from obj, with the first type"""
        # FIXME: unknown type
        for subtype in self.node_type:
            return subtype.extract_tags(obj)
        return {}

    def descriptive_name(self):
        """Return the descriptive names of the types, joined with ``or``."""
        start, remain = self.node_type[0], self.node_type[1:]
        build_str = [start.descriptive_name()]
        for x in remain:
            field = f"or {x.descriptive_name()}"
            build_str.append(field)
        return " ".join(build_str)

    def _datatype_load(self, obj):
        """Load `obj` with the first of the types that can load it.

        Raises
        ------
        TypeError
            If all the types fail with KeyError.
        """
        import numina.store

        faillures = []
        for subtype in self.node_type:
            try:
                return numina.store.load(subtype, obj)
            except KeyError:
                faillures.append(subtype)
        else:
            msg = f"types {faillures} cannot load 'obj'"
            raise TypeError(msg)
