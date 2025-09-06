#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""A representation of a hardware device"""

import inspect
import typing

from .device import DeviceBase


class HWDevice(DeviceBase):

    def init_config_info(self) -> dict[str, typing.Any]:
        return dict(name=self.name)

    def get_property_names(self) -> list[str]:
        result = []
        for key in dir(self.__class__):
            if key.startswith("_"):
                continue
            attr = getattr(self.__class__, key, None)
            if inspect.isdatadescriptor(attr):
                result.append(key)
        return result
