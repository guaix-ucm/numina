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

'''
Recipe requirements
'''



class Requirement(object):
    '''Requirements of Recipes
    
        :param soft: Make the Requirement soft
    
    '''
    def __init__(self, name, value, description, soft=False):
        self.name = name
        self.value = value
        self.description = description
        self.soft = soft

class Parameter(Requirement):
    def __init__(self, name, value, description, soft=False):
        super(Parameter, self).__init__(name, value, description, soft)
