#
# Copyright 2011-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""Modify how to query results in the storage backend"""


import logging


class QueryModifier(object):
    pass


class Result(QueryModifier):
    def __init__(self, mode_field, node=None, ignore_fail=False):
        from numina.types.frame import DataFrameType

        super(Result, self).__init__()

        self.mode_field = mode_field
        self.node = node
        self.ignore_fail = ignore_fail
        self.result_type = DataFrameType()
        splitm = mode_field.split('.')
        lm = len(splitm)
        if lm == 1:
            mode = None
            field = mode_field
        elif lm == 2:
            mode = splitm[0]
            field = splitm[1]
        else:
            raise ValueError('malformed mode_field %s' % mode_field)
        self.mode = mode
        self.field = field


class Ignore(QueryModifier):
    """Ignore this parameter"""
    pass


def basic_mode_builder(mode, partial_ob, backend, options=None):
    logger = logging.getLogger(__name__)

    logger.debug("builder for mode='%s'", mode.name)

    if isinstance(options, Result):
        result_type = options.result_type
        name = 'relative_result'
        logger.debug('query, children id: %s', partial_ob.children)
        val = backend.search_result_relative(name, result_type, partial_ob, result_desc=options)
        if val is None:
            logger.debug('query, children id: %s, no result')
        for r in val:
            partial_ob.results[r.id] = r.content

    return partial_ob