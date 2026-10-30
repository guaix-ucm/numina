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
    """Robot arm of the test instrument, with an angle and an active flag.

    Parameters
    ----------
    name
        Name of the device.
    model
        Model of the robot, an element of the ``ypos`` list of the robot model.
    idx
        Index of the robot, used in the names of the header keywords.
    active : bool, optional
        Whether the robot is active.
    parent
        Parent device.
    """

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
        """Read the angle and the active flag from the keywords ``Rnn_ANGL`` and ``Rnn_ACTV`` of ``hdr``."""
        with self.state.managed_configure():
            self.angle = hdr.get(f"R{self.idx:02d}_ANGL", 0)
            self.active = hdr.get(f"R{self.idx:02d}_ACTV", True)

    @property
    def angle(self):
        """Angle of the robot, that can be set only if it is active or being configured."""
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
        """Whether the robot is active."""
        return self.active_

    @active.setter
    def active(self, value):
        self.active_ = value


class RobotPos(HWDevice):
    """Positioner with a ``Robot`` child for each element of the ``ypos`` list of the robot model.

    Parameters
    ----------
    name
        Name of the device.
    robot_model : dict
        Robot model, with the list ``ypos``.
    origin
        Origin of the configuration.
    parent
        Parent device.
    """

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
        """Create the device with the robot model in the value ``robotmodel`` of ``setup``."""

        robot_model = setup.values["robotmodel"]
        obj = cls.__new__(cls)
        obj.__init__(comp_id, robot_model, origin, parent)
        return obj
