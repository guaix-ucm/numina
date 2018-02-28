#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
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
