#
# Copyright 2008-2011 Universidad Complutense de Madrid
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

'''Sample numina recipe.'''

import logging

from numina.recipes import RecipeBase, RecipeError
import numina.qa as QA

_logger = logging.getLogger("numina.recipes")

class Sample(RecipeBase):
    '''Sample recipe.'''
    
    capabilities = ['sample']
    instrument = ['sample']
    
    __version__ = 1
    
    def __init__(self, param, run):
        super(Sample, self).__init__(param, run)
        
    def run(self):
        if self.current == 0:
            raise RecipeError('Intended')
        return {'qa': QA.UNKNOWN,
                'result': 1}
        
if __name__ == '__main__':
    import tempfile 
    
    from numina.user import main
    from numina.jsonserializer import to_json 
    import json

    p = {'recipe': {'run': {'repeat': 2, 'mode': 'sample', 'instrument': 'sample'}}}   
          
    f = tempfile.NamedTemporaryFile(prefix='tmp', suffix='numina', delete=False)
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
            
    main(['--module', 'numina.recipes', '--run', f.name])
    
