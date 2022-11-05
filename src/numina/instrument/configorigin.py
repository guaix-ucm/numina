#
# Copyright 2019-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Description of the origin of different configurations"""


import dateutil
import uuid as uuidmod

# Try to use isoparse ISO-8601, if not available
# use generic parser
try:
    isoparse = dateutil.parser.isoparse
except AttributeError:
    isoparse = dateutil.parser.parse


class UndefinedOrigin(object):
    """Origin not defined"""
    def is_valid_date(self, cdate):
        return True


class ElementOrigin(object):
    """Description of the origin of a particular configuration"""
    def __init__(self, name, uuid, date_start=None, date_end=None, description=""):
        self.name = name

        if isinstance(uuid, str):
            self.uuid = uuidmod.UUID(uuid)

        self.date_start = self._isoparse(date_start)
        self.date_end = self._isoparse(date_end)

        self.description = description

    @staticmethod
    def _isoparse(date):
        if isinstance(date, str):
            return isoparse(date)
        else:
            return date

    def is_valid_date(self, cdate):
        """Check if the element if valid for a given date"""

        if cdate is None:
            return True

        cdate = self._isoparse(cdate)

        if self.date_start:
            after_start = cdate >= self.date_start
        else:
            after_start = True

        if self.date_end:
            before_end = cdate < self.date_end
        else:
            before_end = True

        if after_start and before_end:
            return True
        else:
            return False

    @classmethod
    def create_from_dict(cls, val):
        return cls.create_from_keys(**val)

    @classmethod
    def create_from_keys(cls, **kwargs):

        # Parse kwargs
        name = kwargs['name']
        desc = kwargs.get('description', "")
        uuid = kwargs['uuid']
        if kwargs['date_start']:
            date_start = isoparse(kwargs['date_start'])
        else:
            date_start = None

        if kwargs['date_end']:
            date_end = isoparse(kwargs['date_end'])
        else:
            date_end = None
        return ElementOrigin(name, uuid, date_start, date_end, desc)
