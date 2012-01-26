#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

import unittest
import tempfile
import os

from numina.diskstorage import store

class DiskStorageTestCase(unittest.TestCase):
    
    def test_register(self):
        
        class MyClass(object):
            pass
        
        @store.register(MyClass) # pylint: disable-msgs=E1101
        def mystore(obj, where):
            where.write('MyClass')
            
        m = MyClass()
        
        f = tempfile.NamedTemporaryFile(delete=False)
        
        store(m, f)
        name = f.name
        f.close()
        
        fo = open(name)
        try:
            line = fo.readline()
            self.assertEqual(line, 'MyClass')
        finally:
            fo.close()
            
        os.remove(name)
        store.unregister(MyClass)

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DiskStorageTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')