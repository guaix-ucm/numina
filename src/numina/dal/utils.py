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


def check_template(template):
    """Check that the template only uses fields in TEMPLATE_FIELDS

    Raises
    ------
    ValueError
        If the template uses other fields
    """
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None and field not in TEMPLATE_FIELDS:
            msg = f"field '{{{field}}}' not allowed in template '{template}', valid fields are {TEMPLATE_FIELDS}"
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
