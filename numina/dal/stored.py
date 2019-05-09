#
# Copyright 2014-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import uuid

from numina.types.qc import QC

from .unserial import unserial


class StoredProduct(object):
    """A product returned from the DAL"""
    def __init__(self, id, content, tags, **kwds):
        self.id = id
        self.content = content
        self.tags = tags


class StoredParameter(object):
    """A parameter returned from the DAL"""
    def __init__(self, content):
        self.content = content


class StoredResult(object):
    """Recover the RecipeResult values stored in the Backend"""

    def __init__(self):
        self.qc = QC.UNKNOWN
        self.uuid = uuid.UUID('00000000-0000-0000-0000-000000000000')

    @classmethod
    def load_data(cls, state):
        obj = cls.__new__(cls)
        obj._from_dict(state)
        return obj

    def _from_dict(self, state):
        self.qc = QC[state.get('qc', 'UNKNOWN')]
        if 'uuid' in state:
            self.uuid = uuid.UUID(state['uuid'])

        values = state.get('values', {})
        if isinstance(values, list):
            values = {o['name']: o['content'] for o in values}
        for key, val in values.items():
            loaded = unserial(val)
            setattr(self, key, loaded)


# A translation of the entries of oblocks
# Notice that this is different to ObservationResult
# that contains the results of the reductions
class ObservingBlock(object):
    def __init__(self, id, instrument, mode, images, children, parent=None, facts=None):
        self.id = id
        self.instrument = instrument
        self.mode = mode
        # only one of files and children can
        # be different from []
        self.images = images
        self.frames = self.images
        self.children = children
        self.parent = parent
        self.facts = facts
