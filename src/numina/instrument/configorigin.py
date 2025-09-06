#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Description of the origin of different configurations"""

from datetime import datetime
import uuid
import typing

from dateutil.parser import isoparse


class UndefinedOrigin:
    """Origin not defined"""

    def is_valid_date(self, _):
        return True


class ElementOrigin:
    """Description of the origin of a particular configuration"""

    def __init__(
        self,
        name: str,
        uuid_in: uuid.UUID | str,
        date_start: str | datetime | None = None,
        date_end: str | datetime | None = None,
        description: str = "",
    ):
        self.name = name

        if isinstance(uuid_in, str):
            self.uuid = uuid.UUID(uuid_in)

        self.date_start = self._isoparse(date_start)
        self.date_end = self._isoparse(date_end)

        self.description = description

    @staticmethod
    def _isoparse(date: str | datetime):
        if isinstance(date, str):
            return isoparse(date)
        else:
            return date

    def is_valid_date(self, cdate: str | None) -> bool:
        """Check if the element is valid for a given date"""

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
    def from_dict(cls, val: dict[str, typing.Any]) -> "ElementOrigin":
        return cls.from_keys(**val)

    @classmethod
    def from_keys(cls, **kwargs) -> "ElementOrigin":

        # Parse kwargs
        name = kwargs["name"]
        desc = kwargs.get("description", "")
        uuid = kwargs["uuid"]
        if kwargs["date_start"]:
            date_start = isoparse(kwargs["date_start"])
        else:
            date_start = None

        if kwargs["date_end"]:
            date_end = isoparse(kwargs["date_end"])
        else:
            date_end = None
        return ElementOrigin(name, uuid, date_start, date_end, desc)
