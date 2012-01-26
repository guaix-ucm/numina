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

import logging

_logger = logging.getLogger('numina.flow')

class FlowError(Exception):
    pass

class Flow(object):
    def __init__(self, ninputs=1, noutputs=1):
        super(Flow, self).__init__()
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
    
    def obtain_tuple(self, arg):
        if isinstance(arg, tuple):
            return arg
        return (arg,)
   
    def execute(self, arg):
        return self._run(arg)

class SerialFlow(Flow):
    def __init__(self, nodeseq):
        # Checking inputs and out puts are correct
        for i,o in zip(nodeseq, nodeseq[1:]):
            if i.noutputs != o.ninputs:
                raise FlowError
        self.nodeseq = nodeseq
        super(SerialFlow, self).__init__(nodeseq[0].ninputs, nodeseq[-1].noutputs)
        
    def __iter__(self):
        return self.nodeseq.__iter__()
    
    def __len__(self):
        return self.nodeseq.__len__()
    
    def __getitem__(self, key):
        return self.nodeseq[key]
    
    def __setitem__(self, key, value):
        self.nodeseq[key] = value

    def _run(self, img):
        for nd in self.nodeseq:
            out = nd(img)
            img = out
        return out

class ParallelFlow(Flow):
    def __init__(self, nodeseq):
        self.nodeseq = nodeseq
        nin = sum((f.ninputs for f in nodeseq), 0)
        nout = sum((f.noutputs for f in nodeseq), 0)
        super(ParallelFlow, self).__init__(nin, nout)
        
        
    def _run(self, img):
        args = self.obtain_tuple(img)
        out = []
        for func, arg in zip(self.nodeseq, args):
            r = func(arg)
            out.append(r)
            
        if any(f.noutputs != 1 for f in self.nodeseq):
            return tuple(item for sublist in out for item in sublist)
        
        return tuple(out)
    
    def __iter__(self):
        return self.nodeseq.__iter__()
    
    def __len__(self):
        return self.nodeseq.__len__()
    
    def __getitem__(self, key):
        return self.nodeseq[key]
    
    def __setitem__(self, key, value):
        self.nodeseq[key] = value
        
        
        
class MixerFlow(Flow):
    def __init__(self, table):
        nin = max(table) + 1
        nout = len(table)
        super(MixerFlow, self).__init__(nin, nout)
        self.table = table
        
    def _run(self, img):
        args = self.obtain_tuple(img)
        assert len(args) == self.ninputs
        
        return tuple(args[idx] for idx in self.table)
