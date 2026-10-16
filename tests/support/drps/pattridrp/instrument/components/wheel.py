#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import enum
import typing

from typing_extensions import Self

if typing.TYPE_CHECKING:
    from numina.instrument.configorigin import ElementOrigin
    from numina.instrument.hwdevice import DeviceBase

from numina.instrument.components.wheel import Wheel


class FilterName(enum.Enum):
    U = 2131
    B = 2134
    V = 2135
    R = 2156
    I = 2158  # noqa
    Z = 2203


class PWheel(Wheel):
    """The particular wheel"""

    def __init__(
        self,
        cid,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(cid, 5, origin=origin, parent=parent)

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
        properties=None,
        setup=None,
    ) -> Self:
        def_filters = []
        if setup is not None:
            def_filters = setup.values["wheel"]["filters"]

        obj = cls.__new__(cls)
        obj.__init__(comp_id, origin=origin)

        for idx, el in enumerate(def_filters):
            obj.put_in_pos(FilterName[el], idx)

        return obj

    def configure_me_with_header(self, hdr):
        if "FILTER" in hdr:
            self.label = hdr["FILTER"]
        else:
            self.position = hdr.get("POS", 1)
        #
        self.is_configured = True

    def my_depends(self):
        return {"filter"}
