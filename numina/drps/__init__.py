
from .drploader import DrpSystemLoaderCached

_system_drps = None

def get_system_drps():
    global _system_drps
    if _system_drps is None:
        _system_drps = DrpSystemLoaderCached()

    return _system_drps