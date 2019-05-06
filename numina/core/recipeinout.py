#
# Copyright 2008-2019 Universidad Complutense de Madrid
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

from six import with_metaclass

from .metaclass import RecipeInputType, RecipeResultType
import numina.store.dump
import numina.types.qc


class RecipeInOut(object):

    def __init__(self, *args, **kwds):
        super(RecipeInOut, self).__init__()
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
            full.append('{0}={1!r}'.format(key, val))
        return '{}({})'.format(sclass, ', '.join(full))

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
            req.validate(val)

        # Run checks defined in __checkers__
        self._run_checks()

    def _run_checks(self):
        checkers = getattr(self, '__checkers__', [])

        for check in checkers:
            check.check(self)


class RecipeInput(with_metaclass(RecipeInputType, RecipeInOut)):
    """RecipeInput base class"""
    pass


class RecipeResultBase(with_metaclass(RecipeResultType, RecipeInOut)):
    """The result of a Recipe."""

    def store_to(self, where):

        saveres = dict(values={})
        saveres_v = saveres['values']
        for key, prod in self.stored().items():
            val = getattr(self, key)
            where.destination = "{}".format(prod.dest)
            saveres_v[key] = numina.store.dump(prod.type, val, where)

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

        saveres['qc'] = self.qc
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
            msg = '{0!r} does not derive from RecipeResult'.format(resultClass)
            raise TypeError(msg)
        self.klass = resultClass

    def __call__(self, klass):
        klass.RecipeResult = self.klass
        return klass


class define_input(object):
    """Recipe decorator."""
    def __init__(self, input_class):
        if not issubclass(input_class, RecipeInput):
            msg = '{0!r} does not derive from RecipeInput'.format(input_class)
            raise TypeError(msg)
        self.klass = input_class

    def __call__(self, klass):
        klass.RecipeInput = self.klass
        return klass


define_requirements = define_input
