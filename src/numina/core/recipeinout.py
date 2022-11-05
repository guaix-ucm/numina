#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""
Recipe inputs and outputs
"""

import uuid
import logging

from .metaclass import RecipeInputType, RecipeResultType
import numina.store.dump
import numina.types.qc


_logger = logging.getLogger(__name__)


class RecipeInOut(object):

    def __init__(self, *args, **kwds):
        super(RecipeInOut, self).__init__()
        # Used to hold set values
        # Use this to avoid infinite recursion
        super(RecipeInOut, self).__setattr__('_numina_desc_val', {})
        # instead of this
        # self._numina_desc_val = {}
        all_msg_errors = []

        # memorize aliases
        super(RecipeInOut, self).__setattr__('_aliases', {})

        for key, req in self.stored().items():
            if req.alias:
                self._aliases[req.alias] = req

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
            full.append(f'{key}={val!r}')
        return f"{sclass}({', '.join(full)})"

    def __getattr__(self, item):
        # This method might be called before _aliases is initialized
        if item in self.__dict__.get('_aliases', {}):
            ref = self.__dict__['_aliases'][item]
            return getattr(self, ref.dest)
        else:
            msg = f"'{self.__class__.__name__}' object has no attribute '{item}'"
            raise AttributeError(msg)

    def __setattr__(self, item, value):
        # This method might be called before _aliases is initialized
        if item in self.__dict__.get('_aliases', {}):
            ref = self.__dict__['_aliases'][item]
            return setattr(self, ref.dest, value)
        else:
            super(RecipeInOut, self).__setattr__(item, value)

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
        return self._numina_desc_val

    @classmethod
    def stored(cls):
        return cls.__numina_stored__

    def validate(self):
        """Validate myself."""

        for key, req in self.stored().items():
            val = getattr(self, key)
            _logger.info('validate %s with a value of %s', req, val)
            try:
                req.validate(val)
                _logger.info('validation passed')
            except Exception as error:
                _logger.warning('validation failed with error %s', error)

        # Run checks defined in __checkers__
        self._run_checks()

    def _run_checks(self):
        checkers = getattr(self, '__checkers__', [])

        for check in checkers:
            check.check(self)

    @classmethod
    def tag_names(cls):
        qfields = set()
        for key, req in cls.stored().items():
            tag_n = req.tag_names()
            qfields.update(tag_n)
        return qfields


class RecipeInput(RecipeInOut, metaclass=RecipeInputType):
    """RecipeInput base class"""
    pass


class RecipeResultBase(RecipeInOut, metaclass=RecipeResultType):
    """The result of a Recipe."""

    def store_to(self, where):

        saveres = dict(values={})
        saveres_v = saveres['values']
        for key, prod in self.stored().items():
            val = getattr(self, key)
            saveres_v[key] = numina.store.dump(prod.type, val, prod.dest)

        return saveres


class RecipeResult(RecipeResultBase):

    def __init__(self, *args, **kwds):

        # Extract QC if available
        self.qc = numina.types.qc.QC.UNKNOWN
        self.uuid = uuid.uuid1()

        # qc is not passed further
        if 'qc' in kwds:
            self.qc = kwds['qc']
            del kwds['qc']

        super(RecipeResult, self).__init__(*args, **kwds)

    def store_to(self, where):
        saveres = super(RecipeResult, self).store_to(where)

        saveres['qc'] = self.qc.name
        saveres['uuid'] = str(self.uuid)
        return saveres

    def time_it(self, time1, time2):
        import numina.types.dataframe as dataframe
        values = self.attrs()
        for k, spec in self.stored().items():
            value = values[k]
            # Store for Images..
            if isinstance(value, dataframe.DataFrame):
                hdul = value.open()
                self.add_computation_time(hdul, time1, time2)

    def add_computation_time(self, img, time1, time2):
        img[0].header['NUMUTC1'] = time1.isoformat()
        img[0].header['NUMUTC2'] = time2.isoformat()
        return img


class define_result(object):
    """Recipe decorator."""
    def __init__(self, resultClass):
        if not issubclass(resultClass, RecipeResult):
            msg = f'{resultClass!r} does not derive from RecipeResult'
            raise TypeError(msg)
        self.klass = resultClass

    def __call__(self, klass):
        klass.RecipeResult = self.klass
        return klass


class define_input(object):
    """Recipe decorator."""
    def __init__(self, input_class):
        if not issubclass(input_class, RecipeInput):
            msg = f'{input_class!r} does not derive from RecipeInput'
            raise TypeError(msg)
        self.klass = input_class

    def __call__(self, klass):
        klass.RecipeInput = self.klass
        return klass


define_requirements = define_input
