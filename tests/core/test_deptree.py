"""Dependency trees, numina.core.deptree is deprecated"""

import importlib
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import numina.core.deptree as deptree


def test_deprecated():
    with pytest.warns(DeprecationWarning, match="numina.core.deptree is deprecated"):
        importlib.reload(deptree)


def test_visit_node():
    leaf = deptree.DepNode("leaf")
    middle = deptree.DepNode("middle", links=[deptree.DepLink(leaf)])
    root = deptree.DepNode("root", links=[deptree.DepLink(middle), deptree.DepLink(leaf, weight=0)])
    visited = []
    deptree.visit_node(root, visited)
    assert visited == ["middle", "leaf"]
