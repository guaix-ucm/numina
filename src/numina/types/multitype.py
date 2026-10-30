#
# Copyright 2017-2019 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import inspect
import warnings

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
        # The type of the value found, only set by the deprecated
        # _query_on_dal. The search of the requirements does not set it.
        self.current_node = None

    def validate(self, obj):
        """Validate `obj` with the current type, or with the first of the types that accepts it.

        Raises
        ------
        numina.exceptions.ValidationError
            If none of the types accepts `obj`.
        """
        if self.current_node:
            return self.current_node.validate(obj)
        else:
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
        """Return whether the current type is a data product, True if there is no current type."""
        if self.current_node:
            return self.current_node.isproduct()
        return True

    def tag_names(self):
        """Return the tag names of the current type, or of all the types."""
        if self.current_node:
            return self.current_node.tag_names()
        else:
            join = []
            for subtype in self.node_type:
                join.extend(subtype.tag_names())
            return join

    def extract_tags(self, obj):
        """Extract tags from obj"""
        if self.current_node:
            return self.current_node.extract_tags(obj)
        else:
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

    def _query_on_dal(self, name, dal, ob, options=None):
        """Deprecated, it is not used by numina and will be removed

        The search is done by Requirement.query_on_dal_rec.
        """
        warnings.warn(
            "MultiType._query_on_dal is deprecated, it is not used by numina and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )

        # Results for subtypes
        results = []
        faillures = []
        for subtype in self.node_type:
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
            self.current_node = subtype
            return result
        else:
            raise nexcep.NoResultFound
