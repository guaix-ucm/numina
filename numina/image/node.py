#
# Copyright 2010-2011 Sergio Pascual
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

import logging
import time

_logger = logging.getLogger('numina.node')

class Node(object):
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
    
    def _run(self, img):
        raise NotImplementedError
    
    def __call__(self, img):
        return self._run(img)

class AdaptorNode(Node):
    def __init__(self, work, ninputs=1, noutputs=1):
        '''work is a function object'''
        super(AdaptorNode, self).__init__(ninputs, noutputs)
        self.work = work

    def _run(self, img):
        return self.work(img)

class IdNode(Node):
    def __init__(self):
        '''Identity'''
        super(IdNode, self).__init__()

    def _run(self, img):
        return img

class OutputSelector(Node):
    def __init__(self, ninputs, indexes):
        noutputs = len(indexes) 
        super(OutputSelector, self).__init__(ninputs, noutputs)
        self.indexes = indexes

    def _run(self, arg):
        res = tuple(ar for idx, ar in enumerate(arg) if idx in self.indexes)
        if len(res) == 1:
            return res[0]
        return res

class Corrector(Node):
    def __init__(self, label=None, mark=True, dtype='float32'):
        super(Corrector, self).__init__()
        self.dtype = dtype
        self.mark = mark
        if not label:
            self.label = ('NUM', 'Numina comment')
        else:
            self.label = label
            
    def __call__(self, img):
        if self.check_if_processed(img):
            _logger.info('%s already processed by %s', img, self)
            return img
        else:
            self._run(img)
        return img

    def check_if_processed(self, img):
        if self.mark and img and img.header.has_key(self.label[0]):
            return True
        return False

    def mark_as_processed(self, img):
        if self.mark:
            img.header.update(self.label[0], time.asctime(), self.label[1])

