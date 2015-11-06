#
# Copyright 2015 Universidad Complutense de Madrid
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


"""Tests for RecipeInput and RecipeResult"""

from numina.core.recipeinout import ErrorRecipeResult
from numina.user.helpers import DiskStorageDefault
import pytest


def test_ErrorRecipeResult_access():

    BB = ErrorRecipeResult("errortype","message","traceback")
    assert BB.errortype is getattr(BB, 'errortype')
    assert BB.message is getattr(BB, 'message')
    assert BB.traceback is getattr(BB, 'traceback')

    assert BB.errortype == 'errortype'
    assert BB.message == 'message'
    assert BB.traceback == 'traceback'


@pytest.mark.usefixtures("numinatmpdir")
def test_ErrorRecipeResult__repr__(tmpdir):

    BB = ErrorRecipeResult("errortype","message","traceback", str(tmpdir)+"/errors.yaml")
    # where = DiskStorageDefault(str(tmpdir))
    print (tmpdir)
    # where.store(BB)
    BB.store()



# def test_class_desc_stored():
#
#     BB = create_input_class()
#
#     stored = BB.stored()
#
#     assert BB.param1 is stored['param1']
#     assert BB.param2 is stored['param2']
#
#
# def test_ins_desc_access():
#
#     BB = create_input_class()
#
#     bb = BB(param1=80)
#
#     values = {'param2': 2, 'param1': 80}
#
#     for key, val in values.items():
#         ival = getattr(bb, key)
#         assert val == ival
#
#     # These values are not stored in __dict__
#     for key in values:
#         assert key not in bb.__dict__
#
#
# def test_ins_attr_access():
#
#     BB = create_input_class()
#
#     bb = BB(param1=80)
#
#     values = {'param2': 2, 'param1': 80}
#
#     for key, val in bb.attrs().items():
#         assert val == values[key]
#
#     # These values are not stored in __dict__
#     for key in values:
#         assert key not in bb.__dict__
#
#
# def test_class_nondesc_access():
#
#     BB = create_input_class()
#     bb = BB(param1=80)
#
#     values = {'param2': 2, 'param1': 80}
#
#     # If we insert a new attribute,it goes to __dict__
#     bb.otherattr = 100
#
#     for key, val in bb.attrs().items():
#         assert val == values[key]
#
#     assert 'otherattr' in bb.__dict__