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

'''Unit test for treedict'''

from  ..treedict import TreeDict

def test_treedict():
    a = TreeDict()
    a['instrument.name'] = 'iname'
    assert(a['instrument.name'] == 'iname')

    de = TreeDict()
    de['val1'] = 'cal1'
    de['val2'] = 2394

    a['instrument.detector'] = de
    assert(a['instrument']['detector']['val2'] == 2394)
    assert(a['instrument.detector.val2'] == 2394)

