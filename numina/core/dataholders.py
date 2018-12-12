#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Recipe requirements
"""

import inspect
import numina.exceptions
import collections

import six

import numina.types.datatype as dt
from .validator import as_list as deco_as_list
from .query import Ignore


class EntryHolder(object):
    def __init__(self, tipo, description, destination, optional,
                 default, choices=None, validation=True):

        super(EntryHolder, self).__init__()

        if tipo is None:
            self.type = dt.NullType()
        elif tipo in [bool, str, int, float, complex, list]:
            self.type = dt.PlainPythonType(ref=tipo())
        elif inspect.isclass(tipo):
            self.type = tipo()
        else:
            self.type = tipo

        self.description = description
        self.optional = optional
        self.dest = destination
        self.default = default
        self.choices = choices
        self.validation = validation

    def __get__(self, instance, owner):
        """Getter of the descriptor protocol."""
        if instance is None:
            return self
        else:
            if self.dest not in instance._numina_desc_val:
                instance._numina_desc_val[self.dest] = self.default_value()

            return instance._numina_desc_val[self.dest]

    def __set__(self, instance, value):
        """Setter of the descriptor protocol."""
        try:
            cval = self.convert(value)
            if self.choices and (cval not in self.choices):
                errmsg = '{} not in {}'.format(cval, self.choices)
                raise numina.exceptions.ValidationError(errmsg)
        except (ValueError, TypeError, numina.exceptions.ValidationError) as err:

            if len(err.args) == 0:
                errmsg = 'UNDEFINED ERROR'
                rem = ()
            else:
                errmsg = err.args[0]
                rem = err.args[1:]

            msg = '"{}": {}'.format(self.dest, errmsg)
            newargs = (msg, ) + rem
            err.args = newargs
            raise

        instance._numina_desc_val[self.dest] = cval

    def convert(self, val):
        return self.type.convert(val)

    def validate(self, val):
        if self.validation:
            return self.type.validate(val)
        return True

    def default_value(self):
        if self.default is not None:
            return self.convert(self.default)
        if self.type.internal_default is not None:
            return self.type.internal_default
        if self.optional:
            return None
        else:
            fmt = 'Required {0!r} of type {1!r} is not defined'
            msg = fmt.format(self.dest, self.type)
            raise ValueError(msg)


class Result(EntryHolder):
    """Result holder for RecipeResult."""
    def __init__(self, ptype, description="", validation=True,
                 destination=None, optional=False, default=None, choices=None):
        super(Result, self).__init__(
            ptype, description, destination=destination, optional=optional,
            default=default, choices=choices, validation=validation
            )

#        if not isinstance(self.type, DataProductType):
#            raise TypeError('type must be of class DataProduct')

    def __repr__(self):
        return 'Result(type=%r, dest=%r)' % (self.type, self.dest)

    def convert(self, val):
        return self.type.convert_out(val)


class Product(Result):
    """Product holder for RecipeResult.

    .. deprecated:: 0.16
            `Product` is replaced by `Result`. It will
            be removed in 1.0
    """

    def __init__(self, ptype, description="", validation=True,
                 destination=None, optional=False, default=None, choices=None):
        super(Product, self).__init__(
            ptype,
            description=description,
            validation=validation,
            destination=destination,
            optional=optional,
            default=default,
            choices=choices
        )

        import warnings
        warnings.warn("The 'Product' class was renamed to 'Result'", DeprecationWarning, stacklevel=2)

    def __repr__(self):
        return 'Product(type=%r, dest=%r)' % (self.type, self.dest)


class Requirement(EntryHolder):
    """Requirement holder for RecipeInput.

    Parameters
    ----------
    rtype : :class:`~numina.types.datatype.DataType` or Type[DataType]
       Object or class repressenting the yype of the requirement,
       it must be a subclass of DataType
    description : str
       Description of the Requirement. The value is used
       by `numina show-recipes` to provide human-readable documentation.
    destination : str, optional
       Name of the field in the RecipeInput object. Overrides the value
       provided by the name of the Requirement variable
    optional : bool, optional
       If `False`, the builder of the RecipeInput must provide a value
       for this Parameter. If `True` (default), the builder can skip
       this Parameter and then the default in `value` is used.
    default : optional
        The value provided by the Requirement if the RecipeInput builder
        does not provide one.
    choices : list of values, optional
        The possible values of the inputs. Any other value will raise
        an exception
    """

    def __init__(self, rtype, description, destination=None, optional=False,
                 default=None, choices=None, validation=True, query_opts=None):
        super(Requirement, self).__init__(
            rtype, description, destination=destination,
            optional=optional, default=default, choices=choices,
            validation=validation
            )

        self.query_opts = query_opts

        self.hidden = False

    def convert(self, val):
        return self.type.convert_in(val)

    def query_options(self):
        return self.query_opts

    def query(self, dal, obsres, options=None):
        # query opts
        if options is not None:
            # Starting with True/False
            perform_query = options
            if not perform_query:
                # we do not perform any query
                return self.default

        if isinstance(self.query_opts, Ignore):
            # we do not perform any query
            return self.default

        if self.dest is None:
            raise ValueError("destination value is not set, "
                             "use the constructor to set destination='value' "
                             "explicitly")

        val = self.type.query(self.dest, dal, obsres, options=self.query_opts)
        return val

    def on_query_not_found(self, notfound):
        self.type.on_query_not_found(notfound)

    def __repr__(self):
        sclass = type(self).__name__
        fmt = ("%s(dest=%r, description='%s', "
               "default=%s, optional=%s, type=%s, choices=%r)"
               )
        return fmt % (sclass, self.dest, self.description, self.default,
                      self.optional, self.type, self.choices)

    def query_constraints(self):
        return self.type.query_constraints()

    def tag_names(self):
        return self.type.tag_names()


def _process_nelem(nlem):
    if nlem is None:
        return False, (None, None)
    if isinstance(nlem, int):
        return True, (nlem, nlem)
    if isinstance(nlem, six.string_types):
        if nlem == '*':
            return True, (0, None)
        if nlem == '+':
            return True, (1, None)

    raise ValueError('value {} is invalid'.format(nlem))


def _recursive_type(value, nmin=None, nmax=None, accept_scalar=True):
    if isinstance(value, (list, tuple)):
        # Continue with contents of list
        if len(value) == 0:
            next_ = dt.AnyType()
        else:
            next_ = value[0]
        final = _recursive_type(next_, accept_scalar=accept_scalar)
        return dt.ListOfType(final, nmin=nmin, nmax=nmax, accept_scalar=accept_scalar)
    elif isinstance(value, (bool, str, int, float, complex)):
        next_ = value
        return dt.PlainPythonType(next_)
    else:
        return value


class Parameter(Requirement):
    """The Recipe requires a plain Python type.

    Parameters
    ----------
    value : plain python type
       Default value of the parameter, the requested type is inferred from
       the type of value.
    description: str
       Description of the parameter. The value is used
       by `numina show-recipes` to provide human-readible documentation.
    destination: str, optional
       Name of the field in the RecipeInput object. Overrides the value
       provided by the name of the Parameter variable
    optional: bool, optional
       If `False`, the builder of the RecipeInput must provide a value
       for this Parameter. If `True` (default), the builder can skip
       this Parameter and then the default in `value` is used.
    choices: list of  plain python type, optional
        The possible values of the inputs. Any other value will raise
        an exception
    validator: callable, optional
        A custom validator for inputs
    accept_scalar: bool, optional
        If `True`, when `value` is a list, scalar value inputs are converted
        to list. If `False` (default), scalar values will raise an exception
        if `value` is a list
    as_list: bool, optional:
        If `True`, consider the internal type a list even if `value` is scalar
        Default is `False`
    nelem: str or int, optional:
        If nelem is '*', the list can contain any number of objects. If is '+',
        the list must contain at least 1 element. With a number, the list must
        contain that number of elements.
    """
    def __init__(self, value, description, destination=None, optional=True,
                 choices=None, validation=True, validator=None,
                 accept_scalar=False,
                 as_list=False, nelem=None,
                 ):

        if nelem is not None:
            decl_list, (nmin, nmax) = _process_nelem(nelem)
        elif as_list:
            decl_list = True
            nmin = nmax = None
        else:
            decl_list = False
            nmin = nmax = None

        is_scalar = not isinstance(value, collections.Iterable)

        if is_scalar and decl_list:
            accept_scalar = True
            value = [value]

        if isinstance(value, (bool, str, int, float, complex, list)):
            default = value
        else:
            default = None

        if validator is None:
            self.custom_validator = None
        elif callable(validator):
            if as_list:
                self.custom_validator = deco_as_list(validator)
            else:
                self.custom_validator = validator
        else:
            raise TypeError('validator must be callable or None')

        mtype = _recursive_type(value,
                                nmin=nmin, nmax=nmax,
                                accept_scalar=accept_scalar
                                )

        super(Parameter, self).__init__(
            mtype, description, destination=destination,
            optional=optional, default=default,
            choices=choices, validation=validation
            )

    def convert(self, val):
        """Convert input values to type values."""
        pre = self.type.convert(val)

        if self.custom_validator is not None:
            post = self.custom_validator(pre)
        else:
            post = pre
        return post

    def validate(self, val):
        """Validate values according to the requirement"""
        if self.validation:
            self.type.validate(val)

            if self.custom_validator is not None:
                self.custom_validator(val)

        return True
