# 
import pkgutil

#
import DF

_eqtypes = {
#'numina.core.products.QualityControlProduct': 'DPK::Something',
'numina.core.products.FrameDataProduct': DF.TYPE_FRAME
}

def dialect_info(obj):
    key = obj.__module__ + '.' + obj.__class__.__name__
    tipo = _eqtypes.get(key, None)
    result = {'gtc': {'fqn': key, 'python': obj.python_type, 'type': tipo}}
    return result

def register(more):
    _eqtypes.update(more)
