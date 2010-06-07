
import logging
import time

_logger = logging.getLogger('numina.node')

class Node(object):
    def __init__(self):
        super(Node, self).__init__()
    
    def __call__(self, img):
        raise NotImplementedError

class SerialNode(Node):
    def __init__(self, nodeseq):
        super(SerialNode, self).__init__()
        self.nodeseq = nodeseq
        
    def __iter__(self):
        return self.nodeseq.__iter__()
    
    def __len__(self):
        return self.nodeseq.__len__()
    
    def __getitem__(self, key):
        return self.nodeseq[key]
    
    def __setitem__(self, key, value):
        self.nodeseq[key] = value

    def __call__(self, img):
        for nd in self.nodeseq:
            out = nd(img)
            img = out
        return out

class AdaptorNode(Node):
    def __init__(self, work):
        '''work is a function object'''
        super(AdaptorNode, self).__init__()
        self.work = work

    def __call__(self, img):
        return self.work(img)

class IdNode(Node):
    def __init__(self):
        '''Identity'''
        super(IdNode, self).__init__()

    def __call__(self, img):
        return img

class ParallelAdaptor(Node):
    def __init__(self, nodeseq):
        super(ParallelAdaptor, self).__init__()
        self.nodeseq = nodeseq
        
    def obtain_tuple(self, arg):
        if isinstance(arg, tuple):
            return arg
        return (arg,)

    def __call__(self, img):
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
        _logger.info('%s already processed by %s', img, self)
        return

    def check_if_processed(self, img):
        if self.mark and img and img.meta.has_key(self.label[0]):
            return True
        return False

    def mark_as_processed(self, img):
        if self.mark:
            img.meta.update(self.label[0], time.asctime(), self.label[1])

