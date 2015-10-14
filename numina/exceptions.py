#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

'''Exceptions for the numina package.'''


class Error(Exception):
    """Base class for exceptions in the numina package."""
    pass


class RecipeError(Error):
    '''A non recoverable problem during recipe execution.'''
    pass


class DetectorElapseError(Error):
    '''Error in the clocking of a Detector.'''
    pass


class DetectorReadoutError(Error):
    '''Error in the readout of a Detector.'''
    pass


class ValidationError(Exception):
    '''Error during validation of Recipe inputs and outputs.'''
    pass


class NoResultFound(Exception):
    """No result found in a DAL query."""
    pass