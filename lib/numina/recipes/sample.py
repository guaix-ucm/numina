#
# Copyright 2008-2010 Sergio Pascual
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

'''Sample numina recipe.'''

import logging

from numina.recipes import RecipeBase, RecipeResult
import numina.qa as QA

_logger = logging.getLogger("numina.recipes")

class Result(RecipeResult):
    '''Result of the sample recipe.'''
    def __init__(self, qa, result, itern):
        super(Result, self).__init__(qa)
        prodkey = 'values%02d.txt' % itern
        self.products[prodkey] = {}
        self.products[prodkey]['result'] = result
        self.products[prodkey]['tables'] = [[0,0], [234.456]]
      
class Recipe(RecipeBase):
    '''Recipe to process data taken in imaging mode.
     
    '''
    
    required_parameters = [
        'niterations'
    ]
    
    def __init__(self, param):
        super(Recipe, self).__init__(param)
        self.repeat = param['niterations']
        self.biter = param['niterations']
        
    def run(self):
        itern =  self.biter - self.repeat     
        return Result(QA.UNKNOWN, self.values['niterations'], itern)
    
if __name__ == '__main__':
    import simplejson as json
    import tempfile
    
    from numina.user import main
    from numina.jsonserializer import to_json 
        
    p = {'niterations': 3}
    
    f = tempfile.NamedTemporaryFile(prefix='tmp', suffix='numina', delete=False)
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
            
    main(['--module', 'numina.recipes', '--run', 'sample', f.name])
    
