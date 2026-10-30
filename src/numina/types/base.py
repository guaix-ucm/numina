#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import inspect
import warnings

from numina.util.parser import parse_arg_line
from numina.datamodel import DataModel


class DataTypeBase:
    """Base class for input/output types of recipes.

    Parameters
    ----------
    datamodel : DataModel or class, optional
        Data model of the instrument, or its class, which is instantiated.
        If None, a ``numina.datamodel.DataModel`` is used.
    query_expr : optional
        Query expression of the type. It replaces the expression built
        from the class attribute ``__tags__``.
    tags : list of str, optional
        Names of the query attributes of the data model used to build the
        query expression. It replaces `query_expr`.
    """

    def __init__(self, *args, **kwds):
        import numina.core.tagexpr as tagexpr

        datamodel = kwds.get("datamodel")

        if datamodel is not None:
            if inspect.isclass(datamodel):
                self.datamodel = datamodel()
            else:
                self.datamodel = datamodel
        else:
            self.datamodel = DataModel()

        my_tag_table = self.datamodel.query_attrs
        self.query_expr = tagexpr.ConstExprTrue

        if hasattr(self, "__tags__"):
            # FIXME:
            if isinstance(self.__tags__, list):
                objtags = [my_tag_table[t] for t in self.__tags__]
            elif isinstance(self.__tags__, dict):
                objtags = [t for t in self.__tags__.values()]
            else:
                raise TypeError("type not supported in tags")

            self.query_expr = tagexpr.query_expr_from_attr(objtags)
        if "query_expr" in kwds:
            self.query_expr = kwds["query_expr"]

        if "tags" in kwds:
            # Create expresion from tags
            objtags = [my_tag_table[t] for t in kwds["tags"]]
            self.query_expr = tagexpr.query_expr_from_attr(objtags)

        self.names_t = self.query_expr.tags()
        self.names_f = self.query_expr.fields()
        self.query_opts = []

    def __getstate__(self):
        """Return an empty state, the attributes of the type are not saved."""
        return {}

    def __setstate__(self, state):
        """Do nothing, the attributes of the type are not restored."""
        pass

    def _datatype_dump(self, obj, where):
        """Return `obj` unchanged, subclasses save it to a file."""
        return obj

    def _datatype_load(self, obj):
        """Return `obj` unchanged, subclasses load it from a file."""
        return obj

    def query_constraints(self):
        """Deprecated, it is not used by numina and will be removed"""
        from numina.core.query import Constraint

        warnings.warn(
            "DataTypeBase.query_constraints is deprecated, it is not used by numina and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )
        return Constraint._create()

    @classmethod
    def isproduct(cls):
        """Check if the DataType is the product of a Recipe"""
        return False

    def __repr__(self):
        sclass = type(self).__name__
        return f"{sclass}()"

    def name(self):
        """Unique name of the datatype"""
        return self.__repr__()

    @classmethod
    def from_name(cls, name):
        """Create a datatype from its name, in the form Class(arg1=val1, arg2=val2)

        Class is the name of the class, or the name of the datatype (see
        :meth:`name`) if it is different, as in the subclasses of DRPs
        that keep the name of a generic type.

        Raises
        ------
        TypeError
            If `name` is not a name of this class.
        """
        # find first (
        fp = name.find("(")
        sp = name.find(")")
        if (fp == -1) or (sp == -1) or (sp < fp):
            # paren not found
            klass = name
            kwargs = {}
        else:
            # parse things between parens
            klass = name[:fp]
            fargs = name[fp + 1 : sp]
            kwargs = parse_arg_line(fargs)

        if klass == cls.__name__:
            return cls(**kwargs)
        obj = cls(**kwargs)
        # the name of the datatype, without the arguments
        if obj.name().split("(", 1)[0] == klass:
            return obj
        raise TypeError(f"{name} is not a name of {cls.__name__}")

    @staticmethod
    def create_db_info():
        """Create metadata structure"""
        result = {}
        result["instrument"] = ""
        result["uuid"] = ""
        result["tags"] = {}
        result["type"] = ""
        result["mode"] = ""
        result["observation_date"] = ""
        result["origin"] = {}
        return result

    def extract_db_info(self, obj, db_info_keys):
        """Extract metadata from serialized file"""
        result = self.create_db_info()
        return result

    def update_meta_info(self):
        """Extract my metadata"""
        result = self.create_db_info()
        return result

    def extract_tags(self, obj):
        """Create metadata structure"""
        return {}

    def tag_names(self):
        """Return the names of the tags in the query expression."""
        return self.names_t
