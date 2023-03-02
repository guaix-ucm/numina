#
# Copyright 2016-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


from numina.instrument.hwdevice import HWDevice
from .signal import Signal


class Carrousel(HWDevice):
    def __init__(self, capacity, name=None, parent=None):
        super(Carrousel, self).__init__(name=name, parent=parent)
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

    def put_in_pos(self, obj, pos):
        if pos >= self._capacity or pos < 0:
            raise ValueError('position greater than capacity or negative')

        self._container[pos] = obj
        self._current = self._container[self._pos]

    def move_to(self, pos):
        if pos >= self._capacity or pos < 0:
            raise ValueError(f'Position {pos:d} out of bounds')

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
            raise ValueError(f'No object named {name}')

    @property
    def position(self):
        return self._pos

    def init_config_info(self):
        info = super(Carrousel, self).init_config_info()
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
        info['selected'] = selected
        return info

    @property
    def label(self):
        if self._current:
            if isinstance(self._current, str):
                lab = self._current
            else:
                lab = self._current.name
        else:
            lab = 'Unknown'

        return lab

    @label.setter
    def label(self, name):
        self.select(name)


class Wheel(Carrousel):
    def __init__(self, capacity, name=None, parent=None):
        super(Wheel, self).__init__(capacity, name=name, parent=parent)

    def turn(self):
        self._pos = (self._pos + 1) % self._capacity
        self._current = self._container[self._pos]
        self.changed.emit(self._pos)
        self.moved.emit(self._pos)
