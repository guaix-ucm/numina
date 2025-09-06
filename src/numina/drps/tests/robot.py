#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


import typing
import warnings

if typing.TYPE_CHECKING:
    from numina.instrument.configorigin import ElementOrigin
    from numina.instrument.hwdevice import DeviceBase

from numina.instrument.hwdevice import HWDevice


class Robot(HWDevice):
    def __init__(
        self,
        name: str,
        model,
        idx: int,
        active=True,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(name, parent=parent)
        self.idx = idx
        self.angle_ = 0
        self.active_ = active
        self.model = model

    def configure_me_with_header(self, hdr):
        with self.state.managed_configure():
            self.angle = hdr.get(f"R{self.idx:02d}_ANGL", 0)
            self.active = hdr.get(f"R{self.idx:02d}_ACTV", True)

    @property
    def angle(self):
        return self.angle_

    @angle.setter
    def angle(self, value: float) -> None:
        if self.active_ or self.state.is_configuring:
            self.angle_ = value
        else:
            # warning, Robot is inactive
            warnings.warn(f"Robot {self.idx} is inactive")
            pass

    @property
    def active(self):
        return self.active_

    @active.setter
    def active(self, value):
        self.active_ = value


class RobotPos(HWDevice):
    def __init__(
        self,
        name: str,
        robot_model,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(name, origin=origin, parent=parent)
        self.robot_model = robot_model
        active = True
        for idx, model in enumerate(robot_model["ypos"]):
            robo = Robot(f"arm_{idx}", model, idx, active=active)
            robo.set_parent(self)

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
        properties=None,
        setup=None,
    ) -> "RobotPos":

        robot_model = setup.values["robotmodel"]
        obj = cls.__new__(cls)
        obj.__init__(comp_id, robot_model, origin, parent)
        return obj
