#
# Copyright 2010-2012 Universidad Complutense de Madrid
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

import abc
import logging

_logger = logging.getLogger('numina.node')

class Node(object):
    '''An elemental operation in a Flow.'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, ninputs=1, noutputs=1):
        super(Node, self).__init__()
        self._nin = ninputs
        self._nout = noutputs
    
    @property
    def ninputs(self):
        return self._nin
    
    @property
    def noutputs(self):
        return self._nout       
    
    @abc.abstractmethod
    def _run(self, img):
        raise NotImplementedError
    
    def __call__(self, img):
        _args = self.obtain_tuple(img)        
        return self._run(img)

    def obtain_tuple(self, arg):
        if isinstance(arg, tuple):
            return arg
        return (arg,)
   
    def execute(self, arg):
        return self._run(arg)

class AdaptorNode(Node):
    '''A :class:`Node` that runs a function.'''
    def __init__(self, work, ninputs=1, noutputs=1):
        '''work is a function object'''
        super(AdaptorNode, self).__init__(ninputs, noutputs)
        self.work = work

    def _run(self, img):
        return self.work(img)

class IdNode(Node):
    '''A Node that returns its inputs.'''
    def __init__(self):
        '''Identity'''
        super(IdNode, self).__init__()

    def _run(self, img):
        return img

class OutputSelector(Node):
    '''A Node that returns part of the results.'''
    def __init__(self, ninputs, indexes):
        noutputs = len(indexes) 
        super(OutputSelector, self).__init__(ninputs, noutputs)
        self.indexes = indexes

    def _run(self, arg):
        res = tuple(ar for idx, ar in enumerate(arg) if idx in self.indexes)
        if len(res) == 1:
            return res[0]
        return res



