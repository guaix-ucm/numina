#
# Copyright 2008-2013 Universidad Complutense de Madrid
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
# along with Numina. If not, see <http://www.gnu.org/licenses/>.
# 

class SimulationBase(object):
    def __init__(self):
        super(SimulationBase, self).__init__()

    command_line = ['--version']

    def simulate(self):
        pass

    def check(self):
        pass

def skip(decorated_class):
    '''Decorate a Simulation class to skip it during tests.'''
    if not issubclass(decorated_class, SimulationBase):
        raise TypeError

    decorated_class.skip = True

    return decorated_class
