#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""
Recipe inputs and outputs
"""

from six import with_metaclass

from .metaclass import RecipeInputType, RecipeResultType
import numina.store.dump
import numina.types.qc


class RecipeInOut(object):

    def __init__(self, *args, **kwds):
        super(RecipeInOut, self).__init__()
        # Used to hold set values
        self._numina_desc_val = {}

        for key, val in kwds.items():
            setattr(self, key, kwds[key])

        self._finalize()

    def __repr__(self):
        sclass = type(self).__name__
        full = []
        for key, val in self.stored().items():
            full.append('%s=%r' % (key, val))
        return '%s(%s)' % (sclass, ', '.join(full))

    def _finalize(self):
        """Access all the instance descriptors

        This wil trigger an exception if a required
        parameter is not set
        """
        all_msg_errors = []

        for key in self.stored():
            try:
                getattr(self, key)
            except ValueError as err:
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


class RecipeResult(with_metaclass(RecipeResultType, RecipeInOut)):
    """The result of a Recipe."""

    def store_to(self, where):

        saveres = {}
        for key, prod in self.stored().items():
            val = getattr(self, key)
            if 'obsid' in where.runinfo:
                where.destination = "{}_{}".format(prod.dest, where.runinfo['obsid'])
            else:
                where.destination = "{}".format(prod.dest)
            saveres[key] = numina.store.dump(prod.type, val, where)

        return saveres


class RecipeResultQC(RecipeResult):
    def __init__(self, *args, **kwds):

        # Extract QC if available
        self.qc = numina.types.qc.QC.UNKNOWN
        if 'qc' in kwds:
            self.qc = kwds['qc']
            del kwds['qc']

        super(RecipeResultQC, self).__init__(*args, **kwds)

    def store_to(self, where):

        saveres = super(RecipeResultQC, self).store_to(where)

        saveres['qc'] = self.qc
        return saveres


class define_result(object):
    """Recipe decorator."""
    def __init__(self, resultClass):
        if not issubclass(resultClass, RecipeResult):
            msg = '%r does not derive from RecipeResult' % resultClass
            raise TypeError(msg)
        self.klass = resultClass

    def __call__(self, klass):
        klass.RecipeResult = self.klass
        return klass


class define_input(object):
    """Recipe decorator."""
    def __init__(self, inputClass):
        if not issubclass(inputClass, RecipeInput):
            fmt = '%r does not derive from RecipeInput'
            msg = fmt % inputClass
            raise TypeError(msg)
        self.klass = inputClass

    def __call__(self, klass):
        klass.RecipeInput = self.klass
        return klass


define_requirements = define_input