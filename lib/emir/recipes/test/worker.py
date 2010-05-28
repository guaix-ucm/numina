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


import Queue
import threading
import logging
import sys

from utils import iterqueue

_logger = logging.getLogger('worker')

class WorkerPool(object):
    def __init__(self, qin=None):
        super(WorkerPool, self).__init__()
        self.pool = []
        if qin is None:
            self.qin = Queue.Queue()
        else:
            self.qin = qin
        self.qout = Queue.Queue()
        self.qerr = Queue.Queue()

    def start(self, node, nthreads=2, daemons=True):
        _logger.debug('creating workers')
        for _i in range(nthreads):
            _logger.debug('worker number %d created', _i)
            self.init_worker(node, daemons)

    def init_worker(self, node, daemons=True):
        wn = Worker(node, self.qin, self.qout, self.qerr)
        nt = threading.Thread(target=wn.run)
        nt.daemon = daemons
        self.pool.append(nt)
        nt.start()


    def stop(self):
        for _idx, nt in enumerate(self.pool):
            self.qin.put(None)
        for nt in self.pool:
            nt.join()
        del self.pool[:]

    def enqueue(self, data):
        self.qin.put_nowait(data)

class Worker(object):
    def __init__(self, node, qin, qout, qerr):
        super(Worker, self).__init__()
        self.node = node
        self.qin = qin
        self.qout = qout
        self.qerr = qerr

    def run(self):
        while True:
            v = self.qin.get()
            # If input is None, we end
            if v is None:
                _logger.debug('worker finished')
                break
            try:
                nv = self.node(v)
            except:
                self.qerr.put(sys.exc_info()[:2])
            else:
                self.qout.put(nv)


def para_map(worker, data, nthreads=4, daemons=True):

    wp = WorkerPool()

    for i in data:
        wp.enqueue(i)

    wp.start(worker, nthreads, daemons)
    wp.stop()

    for i in iterqueue(wp.qerr):
        print i

    result = []
    for i in iterqueue(wp.qout):
        result.append(i)

    return result
