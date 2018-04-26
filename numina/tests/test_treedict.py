#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
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

