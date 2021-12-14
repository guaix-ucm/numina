#
# Copyright 2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


from numina.instrument.hwdevice import HWDevice


class Robot(HWDevice):
    def __init__(self, name, model, parent=None, active=True):
        super(Robot, self).__init__(name, parent=parent)
        self.angle_ = 0
        self.active_ = active
        self.model = model

    @property
    def angle(self):
        return self.angle_

    @angle.setter
    def angle(self, value):
        if self.active_:
            self.angle_ = value
        else:
            # warning, Robot is inactive
            pass

    @property
    def active(self):
        return self.active_

    @active.setter
    def active(self, value):
        self.active_ = value


class RobotPos(HWDevice):
    def __init__(self, name, robotmodel, parent=None):
        super(RobotPos, self).__init__(name, parent=parent)
        self.robotmodel = robotmodel

        model = None
        active = True
        for i in range(1, 10 + 1):
            robo = Robot(f"arm_{i}", model, active=active)
            robo.set_parent(self)

    # Methods added to read from JSON
    @classmethod
    def init_args(self, name, setup_objects):
        robotmodel = setup_objects['robotmodel']
        return (name, robotmodel), {}
