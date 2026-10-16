#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Utilities for DAL"""

import string

# Fields that can be used in the templates of the names
# of the work and results directories, and of the task and result files
TEMPLATE_FIELDS = ("obsid", "taskid")

# Fields that can be used in the template of the names of the
# products of the results: the name of the field of the result
# and the uuid of the product (or of the result)
PRODUCT_TEMPLATE_FIELDS = ("name", "uuid", "obsid", "taskid")


def check_template(template, fields=TEMPLATE_FIELDS):
    """Check that the template only uses fields in `fields`

    Raises
    ------
    ValueError
        If the template uses other fields
    """
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None and field not in fields:
            msg = f"field '{{{field}}}' not allowed in template '{template}', valid fields are {fields}"
            raise ValueError(msg)
    return template


def fill_template(template, obsid, taskid):
    """Fill a template of the names of directories and files"""
    return template.format(obsid=obsid, taskid=taskid)


def tags_are_valid(subset, superset):
    """Validate tags"""
    for key, val in subset.items():
        if key in superset and superset[key] != val:
            return False
    return True
