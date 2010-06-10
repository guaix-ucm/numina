
import logging

import node

_logger = logging.getLogger('numina.flow')


class SerialFlow(node.Node):
    def __init__(self, nodeseq):
        super(SerialFlow, self).__init__()
        self.nodeseq = nodeseq
        
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

class ParallelFlow(node.Node):
    def __init__(self, nodeseq):
        super(ParallelFlow, self).__init__()
        self.nodeseq = nodeseq
        
    def obtain_tuple(self, arg):
        if isinstance(arg, tuple):
            return arg
        return (arg,)

    def _run(self, img):
        args = self.obtain_tuple(img)
        result = tuple(func(arg) for func, arg in zip(self.nodeseq, args))
        return result
    
    def __iter__(self):
        return self.nodeseq.__iter__()
    
    def __len__(self):
        return self.nodeseq.__len__()
    
    def __getitem__(self, key):
        return self.nodeseq[key]
    
    def __setitem__(self, key, value):
        self.nodeseq[key] = value
