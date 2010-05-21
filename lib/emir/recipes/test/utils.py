
import Queue

def iterqueue(qu):
    try:
        while True:
            yield qu.get_nowait()
    except Queue.Empty:
        raise StopIteration