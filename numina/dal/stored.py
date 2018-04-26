#
# Copyright 2014-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


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
