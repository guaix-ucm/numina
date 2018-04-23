#
# Copyright 2008-2017 Universidad Complutense de Madrid
#
# This file is part of PyEmir
#
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
#

"""Double check that __setstate__ and __getstate__ methods work properly."""

import importlib


def check_setstate_getstate(self, json_file_name):
    class_name = self.__class__.__name__

    ClassName = getattr(importlib.import_module('emirdrp.products'),
                        class_name)

    # concatenate __getstate__ and __setstate__
    instance = ClassName()
    instance.__setstate__(self.__getstate__())
    instance.writeto(json_file_name + '_bis')

    # load data from input JSON file and save them again
    instance_bis = ClassName._datatype_load(json_file_name)
    instance_bis.writeto(json_file_name + '_bis2')
