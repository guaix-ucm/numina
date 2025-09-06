#
# Copyright 2016-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import typing

from typing_extensions import Self

if typing.TYPE_CHECKING:
    from numina.instrument.configorigin import ElementOrigin
    from numina.instrument.hwdevice import DeviceBase


from numina.instrument.hwdevice import HWDevice
from numina.instrument.signal import Signal


class Carrousel(HWDevice):
    def __init__(
        self,
        cid,
        capacity: int,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(name=cid, origin=origin, parent=parent)
        # Container is empty
        self._container = [None] * capacity
        self._capacity = capacity
        self._pos = 0
        # object in the current position
        self._current = self._container[self._pos]

        # signals
        self.changed = Signal()
        self.moved = Signal()

    def current(self):
        return self._current

    def pos(self):
        return self._pos

    def put_in_pos(self, obj, pos: int):
        if pos >= self._capacity or pos < 0:
            raise ValueError("position greater than capacity or negative")

        self._container[pos] = obj
        self._current = self._container[self._pos]

    def move_to(self, pos: int):
        if pos >= self._capacity or pos < 0:
            raise ValueError(f"Position {pos:d} out of bounds")

        if pos != self._pos:
            self._pos = pos
            self._current = self._container[self._pos]
            self.changed.emit(self._pos)
        self.moved.emit(self._pos)

    def select(self, name):
        # find pos of object with name
        for idx, item in enumerate(self._container):
            if item:
                if isinstance(item, str):
                    if item == name:
                        return self.move_to(idx)
                elif item.name == name:
                    return self.move_to(idx)
                else:
                    pass
        else:
            raise ValueError(f"No object named {name}")

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, pos: int):
        self.move_to(pos)

    def init_config_info(self):
        info = super().init_config_info()
        if self._current:
            if isinstance(self._current, str):
                selected = self._current
            else:
                try:
                    selected = self._current.config_info()
                except AttributeError:
                    selected = self.label
        else:
            selected = self.label
        info["selected"] = selected
        return info

    @property
    def label(self):
        if self._current:
            if isinstance(self._current, str):
                lab = self._current
            else:
                lab = self._current.name
        else:
            lab = "Unknown"

        return lab

    @label.setter
    def label(self, name):
        self.select(name)

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
        capacity = 1
        if setup is not None:
            capacity = setup.values["capacity"]

        obj = cls.__new__(cls)
        obj.__init__(comp_id, capacity, origin=origin, parent=parent)
        return obj


class Wheel(Carrousel):
    def __init__(
        self,
        cid,
        capacity,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(cid, capacity, origin=origin, parent=parent)

    def turn(self):
        self._pos = (self._pos + 1) % self._capacity
        self._current = self._container[self._pos]
        self.changed.emit(self._pos)
        self.moved.emit(self._pos)
