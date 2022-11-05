#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Validator decorator
"""

from functools import wraps

from numina.exceptions import ValidationError


def validate(method):
    """Decorate run method, inputs and outputs are validated"""

    @wraps(method)
    def mod_run(self, rinput):
        self.validate_input(rinput)
        #
        result = method(self, rinput)
        #
        self.validate_result(result)
        return result

    return mod_run


def only_positive(value):
    """Validation error is value is negative"""
    if value < 0:
        raise ValidationError("must be >= 0")
    return value


def as_list(callable):
    """Convert a scalar validator in a list validator"""
    @wraps(callable)
    def wrapper(value_iter):
        return [callable(value) for value in value_iter]

    return wrapper


def range_validator(minval=None, maxval=None):
    """Generates a function that validates that a number is within range

    Parameters
    ==========
    minval: numeric, optional:
        Values strictly lesser than `minval` are rejected
    maxval: numeric, optional:
        Values strictly greater than `maxval` are rejected

    Returns
    =======
    A function that returns values if are in the range and raises
    ValidationError is the values are outside the range

    """
    def checker_func(value):
        if minval is not None and value < minval:
            msg = f"must be >= {minval}"
            raise ValidationError(msg)
        if maxval is not None and value > maxval:
            msg = f"must be <= {maxval}"
            raise ValidationError(msg)
        return value

    return checker_func
