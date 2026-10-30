#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import inspect
import collections.abc

from numina.exceptions import ValidationError
from .base import DataTypeBase
from .typedialect import dialect_info


class DataType(DataTypeBase):
    """Base class for input/output types of recipes.

    Parameters
    ----------
    ptype : type
        Python type of the values.
    node_type : optional
        Type of the elements, in types that contain other types.
    default : optional
        Default value.
    **kwds
        Passed to ``DataTypeBase``.
    """

    def __init__(self, ptype, node_type=None, default=None, **kwds):
        super().__init__(**kwds)
        self.node_type = node_type
        self.internal_type = ptype
        self.internal_dialect = dialect_info(self)
        self.internal_default = default
        self.internal_scalar = True
        self.multi_query = False

    def convert(self, obj):
        """Basic conversion to internal type

        This method is intended to be redefined by subclasses
        """
        return obj

    def convert_in(self, obj):
        """Basic conversion to internal type of inputs.

        This method is intended to be redefined by subclasses
        """
        return self.convert(obj)

    def convert_out(self, obj):
        """Basic conversion to internal type of outputs.

        This method is intended to be redefined by subclasses
        """
        return self.convert(obj)

    def validate(self, obj):
        """Validate convertibility to internal representation

        Returns
        -------
        bool
          True if 'obj' matches the data type

        Raises
        -------
        ValidationError
            If the validation fails

        """
        if not isinstance(obj, self.internal_type):
            raise ValidationError(obj, self.internal_type)
        return True

    def add_dialect_info(self, dialect, tipo):
        """Add the description of the type in the dialect `dialect`, and return it.

        Parameters
        ----------
        dialect : str
            Name of the dialect, for example ``"gtc"``.
        tipo
            The type in the dialect, for example ``DF.TYPE_FRAME``.

        Returns
        -------
        dict
            The fully qualified name of the class (``"fqn"``), the Python
            type (``"python"``) and `tipo` (``"type"``).
        """
        key = self.__module__ + "." + self.__class__.__name__
        result = {"fqn": key, "python": self.internal_type, "type": tipo}
        self.internal_dialect[dialect] = result
        return result

    def descriptive_name(self):
        """Return the name of the type shown to the user, by default its ``name()``."""
        return self.name()


class AutoDataType(DataType):
    """Data type for types that are its own python type"""

    def __init__(self, *args, **kwargs):
        super().__init__(ptype=self.__class__, **kwargs)


class AnyType(DataType):
    """Type representing anything"""

    def __init__(self):
        super().__init__(ptype=self.__class__)

    def convert(self, obj):
        """Return `obj` unchanged."""
        return obj

    def validate(self, obj):
        """Accept any object, return True."""
        return True


class NullType(DataType):
    """Data type for None."""

    def __init__(self):
        super().__init__(type(None))

    def convert(self, obj):
        """Return None."""
        return None

    def validate(self, obj):
        """Check that `obj` is None, raise ValidationError if it is not."""
        if obj is not None:
            raise ValidationError
        return True


class PlainPythonType(DataType):
    """Data type for Python basic types.

    Parameters
    ----------
    ref : optional
        A value of the type, for example ``0`` or ``""``. Its type is the
        Python type of the values, and the value built by the type without
        arguments is the default value. If None, the type is ``NoneType``
        and there is no default value.
    validator : callable, optional
        Function called with the converted value, its return value is the
        result of the conversion.

    Raises
    ------
    TypeError
        If `validator` is not callable or None.
    """

    def __init__(self, ref=None, validator=None):
        stype = type(ref)
        if ref is None:
            default = None
        else:
            default = stype()

        if validator is None:
            self.custom_validator = None
        elif callable(validator):
            self.custom_validator = validator
        else:
            raise TypeError("validator must be callable or None")

        super().__init__(stype, default=default)

    def convert(self, obj):
        """Convert `obj` to the Python type and apply the validator, if any."""
        pre = self.internal_type(obj)

        if self.custom_validator is not None:
            m = self.custom_validator(pre)
            return m

        return pre

    def __str__(self):
        sclass = type(self).__name__
        return f"{sclass}[{self.internal_type}]"


class ListOfType(DataType):
    """Data type for lists of other types.

    Parameters
    ----------
    ref : DataType or class
        Type of the elements, or its class, which is instantiated.
    default : optional
        Default value.
    index : int, optional
        Index of the first element, used in the names of the files
        when the elements are saved.
    nmin, nmax : int, optional
        Minimum and maximum number of elements.
    accept_scalar : bool, optional
        If True, a value that is not iterable is converted to a list of one element.
    multi_query : bool, optional
        If True, the list is obtained with a query to the DAL for each set
        of tags of the observation result. If False, with a single query
        that returns the list. If None, it is True if the elements are data
        products.
    """

    def __init__(self, ref, default=None, index=0, nmin=None, nmax=None, accept_scalar=False, multi_query=None):
        from numina.core.validator import range_validator

        stype = list
        if inspect.isclass(ref):
            node_type = ref()
        else:
            node_type = ref
        super().__init__(stype, node_type=node_type, default=default)
        self.internal_scalar = False
        self.index = index
        self.nmin = nmin
        self.nmax = nmax
        self.accept_scalar = accept_scalar
        self.len_validator = range_validator(minval=nmin, maxval=nmax)

        # If multi_query is True, we perform N queries concatenated into a list
        # If multi_query is False, we perform 1 query to obtain directly a list
        if multi_query is None:
            self.multi_query = self.node_type.isproduct()
        else:
            self.multi_query = multi_query

    def convert(self, obj):
        """Convert each element with the type of the elements, and check the number of elements.

        Raises
        ------
        TypeError
            If `obj` is not iterable and scalars are not accepted.
        numina.exceptions.ValidationError
            If the number of elements is out of range.
        """
        if not isinstance(obj, collections.abc.Iterable):
            if self.accept_scalar:
                obj = [obj]
            else:
                raise TypeError(
                    "The object received should be iterable" " or the type modified to accept scalar values"
                )

        result = [self.node_type.convert(o) for o in obj]
        self.len_validator(len(result))
        return result

    def validate(self, obj):
        """Validate each element with the type of the elements, and check the number of elements."""
        for o in obj:
            self.node_type.validate(o)
        self.len_validator(len(obj))
        return True

    def _datatype_dump(self, objs, where):
        """Save each element with the type of the elements, adding its index to `where`, return the list."""
        result = []
        old_dest = where
        for idx, obj in enumerate(objs, start=self.index):
            n_where = f"{old_dest}{idx}"
            res = self.node_type._datatype_dump(obj, n_where)
            result.append(res)
        return result

    def _datatype_load(self, objs):
        """Load each element with the type of the elements, return the list."""
        if not isinstance(objs, collections.abc.Iterable):
            if self.accept_scalar:
                objs = [objs]
            else:
                raise TypeError(
                    "The object received should be iterable" " or the type modified to accept scalar values"
                )
        import numina.store

        return [numina.store.load(self.node_type, obj) for obj in objs]

    def __str__(self):
        sclass = type(self).__name__
        return f"{sclass}[{self.node_type}]"
