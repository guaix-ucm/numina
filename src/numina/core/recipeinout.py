#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""
Recipe inputs and outputs
"""

import uuid
import logging
import warnings

import numina.exceptions

from .dataholders import Requirement, Result
import numina.store.dump
import numina.types.qc
from numina.types.datatype import ListOfType

_logger = logging.getLogger(__name__)


class _Alias:
    """Alternative name of a field of a RecipeInOut"""

    def __init__(self, dest):
        self.dest = dest

    def __get__(self, instance, owner):
        """Return the value of the field ``dest``."""
        if instance is None:
            return self
        return getattr(instance, self.dest)

    def __set__(self, instance, value):
        """Set the value of the field ``dest``."""
        setattr(instance, self.dest, value)


def product_uuid(ptype, value):
    """The uuid of a product of a result, None if it has none

    The uuid of an image is read with the data model of its type, as the
    registry of reductions does; other values use their attribute uuid.
    """
    import numina.types.dataframe as dataframe

    if value is None:
        return None
    if isinstance(value, dataframe.DataFrame):
        if value.frame is None:
            return None
        datamodel = getattr(ptype, "datamodel", None)
        if datamodel is None:
            uuid_ = value.frame[0].header.get("UUID")
        else:
            uuid_ = datamodel.extractor_map["fits"].extract("uuid", value.frame)
    else:
        uuid_ = getattr(value, "uuid", None)
    return None if uuid_ in (None, "") else str(uuid_)


class RecipeInOut:
    """Base class of the inputs and the results of the recipes.

    The fields are the class attributes of type ``_numina_field_type``,
    stored by name, with those of the base classes, in ``__numina_stored__``.
    A field with a destination is renamed to it, and a field with an alias
    can be accessed also by the alias.
    """

    _numina_field_type = None

    def __init_subclass__(cls, **kwargs):
        """Collect the fields of the class and its bases, and create the aliases."""
        super().__init_subclass__(**kwargs)

        stored = {}
        for base in cls.__bases__:
            stored.update(getattr(base, "__numina_stored__", {}))

        field_type = cls._numina_field_type
        for name, value in list(vars(cls).items()):
            if field_type is not None and isinstance(value, field_type):
                # the destination is set by EntryHolder.__set_name__
                if value.dest != name:
                    delattr(cls, name)
                    setattr(cls, value.dest, value)
                stored[value.dest] = value

        cls.__numina_stored__ = stored

        for value in stored.values():
            # a field has priority over an alias with the same name
            if value.alias and value.alias not in stored and value.alias not in vars(cls):
                setattr(cls, value.alias, _Alias(value.dest))

    def __init__(self, *args, **kwds):
        """Set the fields from the keyword arguments, the positional arguments are ignored.

        Raises ValueError with the list of the errors of all the fields,
        including the required fields without value.
        """
        super().__init__()
        # Used to hold set values
        self._numina_desc_val = {}
        all_msg_errors = []

        for key, val in kwds.items():
            try:
                setattr(self, key, kwds[key])
            except (ValueError, TypeError) as err:
                all_msg_errors.append(err.args[0])

        self._finalize(all_msg_errors)

    def __repr__(self):
        sclass = type(self).__name__
        full = []
        for key, val in self.stored().items():
            full.append(f"{key}={val!r}")
        return f"{sclass}({', '.join(full)})"

    def _finalize(self, all_msg_errors=None):
        """Access all the instance descriptors

        This wil trigger an exception if a required
        parameter is not set
        """
        if all_msg_errors is None:
            all_msg_errors = []

        for key in self.stored():
            try:
                getattr(self, key)
            except (ValueError, TypeError) as err:
                all_msg_errors.append(err.args[0])

        # Raises a list of all the missing entries
        if all_msg_errors:
            raise ValueError(all_msg_errors)

    def attrs(self):
        """Return the values of the fields, by name."""
        return self._numina_desc_val

    @classmethod
    def stored(cls):
        """Return the fields of the class, by name."""
        return cls.__numina_stored__

    def validate(self):
        """Validate the values of the fields.

        Each value is validated by its field, and then the checks of the
        class attribute ``__checkers__``, a list of objects with a method
        ``check(recipe_inout)``, that can validate the values together.
        The optional fields without value are not validated.

        Raises
        ------
        numina.exceptions.ValidationError
            With the errors of all the fields and checks.
        """

        errors = []
        for key, req in self.stored().items():
            val = getattr(self, key)
            if val is None and req.optional:
                continue
            _logger.debug("validate %s with a value of %s", key, val)
            try:
                req.validate(val)
            except Exception as error:
                errors.append(f"{key}: {_error_message(error)}")

        # Run checks defined in __checkers__
        errors.extend(self._run_checks())

        if errors:
            for error in errors:
                _logger.error("validation failed, %s", error)
            raise numina.exceptions.ValidationError(f"invalid {type(self).__name__}: " + "; ".join(errors))

    def _run_checks(self):
        """Run the checks of __checkers__, return the list of errors"""
        errors = []
        checkers = getattr(self, "__checkers__", [])

        for check in checkers:
            try:
                check.check(self)
            except Exception as error:
                errors.append(f"{type(check).__name__}: {_error_message(error)}")
        return errors

    @classmethod
    def tag_names(cls):
        """Return the set of the names of the tags of the types of the fields."""
        qfields = set()
        for key, req in cls.stored().items():
            tag_n = req.tag_names()
            qfields.update(tag_n)
        return qfields


class RecipeInput(RecipeInOut):
    """RecipeInput base class"""

    _numina_field_type = Requirement


class RecipeResultBase(RecipeInOut):
    """The result of a Recipe."""

    _numina_field_type = Result

    def store_to(self, where, product_tmpl="{name}", fields=None):
        """Save the values of the result, return their serialized form

        Parameters
        ----------
        where
            Not used.
        product_tmpl : str
            Template of the base of the names of the files of the
            products, with the fields {name} (the name of the field
            of the result), {uuid} (the uuid of the product, or of the
            result if the product has none) and the values in `fields`.
            The default, {name}, gives master_bias.fits for the field
            master_bias.
        fields : dict, optional
            Other values of the template, such as obsid and taskid.
        """
        fields = fields or {}
        saveres = dict(values={})
        saveres_v = saveres["values"]
        for key, prod in self.stored().items():
            val = getattr(self, key)
            saveres_v[key] = self._store_value(prod, val, product_tmpl, fields)

        return saveres

    def _store_value(self, prod, val, product_tmpl, fields):
        """Save one value of the result, with a name from the template"""
        if product_tmpl == "{name}":
            return numina.store.dump(prod.type, val, prod.dest)

        result_uuid = getattr(self, "uuid", None)
        if isinstance(prod.type, ListOfType) and val is not None:
            # each element has its own name, with its index after the name
            dumped = []
            for idx, item in enumerate(val, start=prod.type.index):
                uuid_ = product_uuid(prod.type.node_type, item) or result_uuid
                base = product_tmpl.format(name=f"{prod.dest}{idx}", uuid=uuid_, **fields)
                dumped.append(numina.store.dump(prod.type.node_type, item, base))
            return dumped

        uuid_ = product_uuid(prod.type, val) or result_uuid
        base = product_tmpl.format(name=prod.dest, uuid=uuid_, **fields)
        return numina.store.dump(prod.type, val, base)


class RecipeResult(RecipeResultBase):
    """The result of a recipe, with its quality control and its uuid.

    Parameters
    ----------
    qc : numina.types.qc.QC, optional
        Quality control of the result, by default ``QC.UNKNOWN``.
    ``**kwds``
        The values of the fields of the result.
    """

    def __init__(self, *args, **kwds):

        # Extract QC if available
        self.qc = numina.types.qc.QC.UNKNOWN
        self.uuid = uuid.uuid1()

        # qc is not passed further
        if "qc" in kwds:
            self.qc = kwds["qc"]
            del kwds["qc"]

        super().__init__(*args, **kwds)

    def store_to(self, where, product_tmpl="{name}", fields=None):
        """Save the values of the result, adding its ``qc`` and ``uuid`` to the serialized form."""
        saveres = super().store_to(where, product_tmpl=product_tmpl, fields=fields)

        saveres["qc"] = self.qc.name
        saveres["uuid"] = str(self.uuid)
        return saveres

    def time_it(self, time1, time2):
        """Write the start and end times of the run in the images of the result."""
        import numina.types.dataframe as dataframe

        values = self.attrs()
        for k, spec in self.stored().items():
            value = values[k]
            # Store for Images..
            if isinstance(value, dataframe.DataFrame):
                hdul = value.open()
                self.add_computation_time(hdul, time1, time2)

    def add_computation_time(self, img, time1, time2):
        """Write `time1` and `time2` in NUMUTC1 and NUMUTC2 of the primary header of `img`, return `img`."""
        img[0].header["NUMUTC1"] = time1.isoformat()
        img[0].header["NUMUTC2"] = time2.isoformat()
        return img


class define_result:
    """Recipe decorator, that sets the result class of the recipe

    .. deprecated:: 0.41
            Declare the results in the recipe, that creates its result
            class. The decorator replaces the result class, and the results
            declared in the recipe and in its base recipes are lost. It
            will be removed in 0.42.
    """

    def __init__(self, resultClass):
        warnings.warn(
            "define_result is deprecated, declare the results in the recipe",
            DeprecationWarning,
            stacklevel=2,
        )
        if not issubclass(resultClass, RecipeResult):
            msg = f"{resultClass!r} does not derive from RecipeResult"
            raise TypeError(msg)
        self.klass = resultClass

    def __call__(self, klass):
        """Set the result class of the recipe `klass`."""
        klass.RecipeResult = self.klass
        return klass


class define_input:
    """Recipe decorator, that sets the input class of the recipe

    .. deprecated:: 0.41
            Declare the requirements in the recipe, that creates its input
            class. The decorator replaces the input class, and the
            requirements declared in the recipe and in its base recipes,
            and the checks of the input class of the base recipe, are lost.
            It will be removed in 0.42.
    """

    def __init__(self, input_class):
        warnings.warn(
            "define_input is deprecated, declare the requirements in the recipe",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_class(input_class)

    def _set_class(self, input_class):
        """Store `input_class`, raise TypeError if it does not derive from RecipeInput."""
        if not issubclass(input_class, RecipeInput):
            msg = f"{input_class!r} does not derive from RecipeInput"
            raise TypeError(msg)
        self.klass = input_class

    def __call__(self, klass):
        """Set the input class of the recipe `klass`."""
        klass.RecipeInput = self.klass
        return klass


class define_requirements(define_input):
    """Deprecated alias of define_input"""

    def __init__(self, input_class):
        warnings.warn(
            "define_requirements is deprecated, declare the requirements in the recipe",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_class(input_class)


def generate_docs(klass):
    """Add documentation to generated classes"""
    import numina.types.datatype

    attrh = "Attributes\n" "----------\n"

    doc = getattr(klass, "__doc__", None)

    if doc is None or doc == "":
        doc = f"{klass.__name__} documentation."

    if len(klass.stored()):
        doc = doc + "\n\n" + attrh

    skeys = sorted(klass.stored().keys())
    for key in skeys:
        y = klass.stored()[key]

        if isinstance(y, Requirement):
            modo = "requirement"
        elif isinstance(y, Result):
            modo = "product"
        else:
            modo = ""
        if y.type.isproduct():
            tipo = y.type.__class__.__name__
        elif isinstance(y.type, numina.types.datatype.PlainPythonType):
            tipo = y.type.internal_type.__name__
        else:
            tipo = y.type.__class__.__name__

        if y.optional:
            if y.default_value():
                modo = f"{modo}, optional, default={y.default}"
            else:
                modo = f"{modo}, optional"

        descript = y.description
        if descript:
            field = f"{key} : {tipo}, {modo}\n {descript}\n"
        else:
            field = f"{key} : {tipo}, {modo}\n"
        doc = doc + field

    klass.__doc__ = doc
    return klass


def _error_message(error):
    """The message of a validation error, with its type if it is not a ValidationError"""
    msg = str(error) or repr(error)
    if isinstance(error, numina.exceptions.ValidationError):
        return msg
    return f"{type(error).__name__}: {msg}"


class TagsChecker:
    """Check that the tags of the products of a recipe input match those of its observation result

    A check for the class attribute ``__checkers__`` of a RecipeInput,
    run by its method ``validate``. The tags of each product are checked
    with the query expression of the type of its requirement, as in the
    automatic selection of products. Without observation result there is
    nothing to check.

    Examples
    --------
    The input of the recipes of a pipeline, with the check::

        class InstrumentRecipeInput(RecipeInput):
            __checkers__ = [TagsChecker()]

        class InstrumentBaseRecipe(BaseRecipe):
            RecipeInput = InstrumentRecipeInput
    """

    def check(self, recipe_input):
        """Raise numina.exceptions.ValidationError if the tags of a product do not match"""
        obresult = getattr(recipe_input, "obresult", None)
        if obresult is None:
            return
        stored = recipe_input.stored()
        for key, value in recipe_input.attrs().items():
            if key == "obresult" or value is None:
                continue
            rtype = stored[key].type
            check_product_tags(key, f"the value of {key}", rtype, rtype.extract_tags(value), obresult.tags)


_MISSING = object()


def check_product_tags(key, what, rtype, tags, ob_tags):
    """Check that the tags of a product match those of an observation result

    The tags are checked with the query expression of `rtype`, the type of
    the requirement `key`, with the tags of the observation result in its
    placeholders. With a type of several types (MultiType), the tags must
    match one of them. Types that are not products are not checked.

    Parameters
    ----------
    key : str
        Name of the requirement.
    what : str
        Description of the product, for the error message.
    rtype : DataType
    tags : dict
        Tags of the product.
    ob_tags : dict
        Tags of the observation result.

    Raises
    ------
    numina.exceptions.ValidationError
        If the tags do not match.
    """
    node_types = getattr(rtype, "node_type", None) or [rtype]
    exprs = [node.query_expr for node in node_types if node.isproduct() and hasattr(node, "query_expr")]
    if not exprs:
        return
    if any(expr.fill_placeholders(**ob_tags).eval(**tags) is True for expr in exprs):
        return
    names = sorted(set().union(*(expr.tags() for expr in exprs)))
    different = [name for name in names if tags.get(name, _MISSING) != ob_tags.get(name, _MISSING)]
    if different:
        product_tags = ", ".join(f"{name}={tags[name]}" if name in tags else f"no tag {name}" for name in different)
        values = ", ".join(f"{name}={ob_tags.get(name)}" for name in different)
        msg = f"requirement {key}: {what} has {product_tags}, the observing block has {values}"
    else:
        msg = f"requirement {key}: the tags of {what}, {tags}, do not match those of the observing block, {ob_tags}"
    raise numina.exceptions.ValidationError(msg)
