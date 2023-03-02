#
# Copyright 2017-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


class DepLink(object):
    """
    Dependency Link
    """
    def __init__(self, node, weight=1):
        self.node = node
        self.weight = weight


class DepNode(object):
    """Dependency Node"""
    def __init__(self, name, links=None):
        self.name = name
        self.links = [] if links is None else links

    def showtree(self):
        visited = []
        # print node.name
        visited.append(self.name)
        visit_node(self, visited, level=1)


def visit_node(node, visited, level=1):
    """
    Visit nodes in a dependency tree
    Parameters
    ----------
    node
    visited
    level

    Returns
    -------

    """
    # filler = 3 * level * "-"
    for l in node.links:
        # print filler, "> link,",level,", w=",l.weight, " dest=", l.node.name
        # print filler, ">", l.node.name
        if l.node.name in visited:
            continue
        else:
            visited.append(l.node.name)
        visit_node(l.node, visited, level+1)
