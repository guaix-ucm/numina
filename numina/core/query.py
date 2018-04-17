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


class Constraint(QueryModifier):
    pass


class ResultOf(QueryModifier):
    def __init__(self, field, node="children", ignore_fail=False,
                 id_field=None):
        from numina.types.frame import DataFrameType

        super(ResultOf, self).__init__()

        self.field = field

        if node not in ['children', 'prev', 'prev-rel']:
            raise ValueError("value '{}' not allowed for node".format(node))

        self.node = node
        if self.node == 'children':
            self.id_field = id_field or "children"
        elif self.node in ['prev', 'prev-rel']:
            self.id_field = id_field or "prev"

        self.ignore_fail = ignore_fail
        self.result_type = DataFrameType()

        splitm = field.split('.')
        lm = len(splitm)
        if lm == 1:
            self.mode = None
            self.attr = field
        elif lm == 2:
            self.mode = splitm[0]
            self.attr = splitm[1]
        else:
            raise ValueError('malformed desc: %s' % field)


class Ignore(QueryModifier):
    """Ignore this parameter"""
    pass


def basic_mode_builder(mode, partial_ob, backend, options=None):
    logger = logging.getLogger(__name__)

    logger.debug("builder for mode='%s'", mode.name)

    if isinstance(options, ResultOf):
        result_type = options.result_type
        name = 'relative_result'
        logger.debug('query, children id: %s', partial_ob.children)
        val = backend.search_result_relative(name, result_type, partial_ob, result_desc=options)
        if val is None:
            logger.debug('query, children id: %s, no result')
        for r in val:
            partial_ob.results[r.id] = r.content

    return partial_ob
