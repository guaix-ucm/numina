#
# Copyright 2011-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Modify how to query results in the storage backend"""

import warnings


class QueryModifier:
    """Base class of the options that modify the query of a requirement.

    They are given in the 'query_opts' of a requirement or in the 'links'
    of a recipe in drp.yaml.
    """


class Constraint(QueryModifier):
    """Deprecated, it is not used by numina and will be removed"""

    def __init__(self):
        warnings.warn(
            "Constraint is deprecated, it is not used by numina and will be removed", DeprecationWarning, stacklevel=2
        )
        super().__init__()

    @classmethod
    def _create(cls):
        """Create an instance without warning, for the deprecated query_constraints"""
        return cls.__new__(cls)


class ResultOf(QueryModifier):
    """Query the result of other observing blocks

    Parameters
    ----------
    field : str
        Field of the result, as 'MODE.field' or 'field'. The mode
        is checked with node='prev' and node='prev-rel', and it is
        required with node='last'.
    node : str, optional
        Observing blocks where the result is searched:

        - 'children': the children of the observing block, one result each.
        - 'prev': the closest previous observing block with a result,
          in the order they were loaded.
        - 'prev-rel': as 'prev', but only among the previous children
          of the same parent; if there is no parent, as 'prev'.
        - 'last': the most recent result of the mode, of any observing
          block, with the same instrument and instrument profile and
          quality control not BAD. It requires a registry of reductions.
    ignore_fail : bool, optional
        With node='children', skip the children without result
        instead of raising NoResultFound.
    id_field : str, optional
        Not used. The children are always read from the 'children'
        field of the observing block.
    """

    def __init__(self, field, node="children", ignore_fail=False, id_field=None):
        from numina.types.frame import DataFrameType

        super().__init__()

        self.field = field

        if node not in ["children", "prev", "prev-rel", "last"]:
            raise ValueError(f"value '{node}' not allowed for node")

        self.node = node
        if self.node == "children":
            self.id_field = id_field or "children"
        elif self.node in ["prev", "prev-rel"]:
            self.id_field = id_field or "prev"
        elif self.node == "last":
            self.id_field = id_field or "last"

        self.ignore_fail = ignore_fail
        self.result_type = DataFrameType()

        splitm = field.split(".")
        lm = len(splitm)
        if lm == 1:
            self.mode = None
            self.attr = field
        elif lm == 2:
            self.mode = splitm[0]
            self.attr = splitm[1]
        else:
            raise ValueError(f"malformed desc: {field}")

        if self.node == "last" and self.mode is None:
            raise ValueError(f"node 'last' requires the mode in the field, as 'MODE.{self.attr}'")


class Ignore(QueryModifier):
    """Ignore this parameter"""

    pass
