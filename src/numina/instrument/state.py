#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import contextlib
import enum
from typing import Generator


from .signal import Signal


class Status(enum.Enum):
    """Enum representing the status of the component"""

    STATUS_ACTIVE = 1
    STATUS_FAILED = 2
    STATUS_RESETTING = 3
    STATUS_CONFIGURING = 4


class Transition(enum.Enum):
    TRANSITION_RESET = 0
    TRANSITION_END_RESET = 1
    TRANSITION_CONFIGURE = 2
    TRANSITION_END_CONFIGURE = 3
    TRANSITION_ERROR = 4


class State:
    def __init__(self):
        self.current = Status.STATUS_ACTIVE
        #
        self.enter_configure = Signal()
        self.exit_configure = Signal()
        self.enter_reset = Signal()
        self.exit_reset = Signal()
        self.enter_active = Signal()
        self.exit_active = Signal()
        self.enter_failed = Signal()
        self.exit_failed = Signal()

    @property
    def is_configuring(self) -> bool:
        return self.current == Status.STATUS_CONFIGURING

    def transition(self, inpt: Transition) -> None:
        match self.current:
            case Status.STATUS_FAILED:
                match inpt:
                    case Transition.TRANSITION_RESET:
                        self.exit_failed.emit()
                        self.current = Status.STATUS_RESETTING
                        self.enter_reset.emit()
                    case _:
                        pass  # No action
            case Status.STATUS_RESETTING:
                match inpt:
                    case Transition.TRANSITION_RESET:
                        pass  # No action
                    case Transition.TRANSITION_END_RESET:
                        self.exit_reset.emit()
                        self.current = Status.STATUS_ACTIVE
                        self.enter_active.emit()
                    case _:
                        self.exit_reset.emit()
                        self.current = Status.STATUS_FAILED
                        self.enter_failed.emit()
            case Status.STATUS_CONFIGURING:
                match inpt:
                    case Transition.TRANSITION_CONFIGURE:
                        pass  # No action
                    case Transition.TRANSITION_END_CONFIGURE:
                        self.current = Status.STATUS_ACTIVE
                        self.exit_configure.emit()
                    case _:
                        self.current = Status.STATUS_FAILED
                        self.enter_failed.emit()
            case Status.STATUS_ACTIVE:
                match inpt:
                    case Transition.TRANSITION_CONFIGURE:
                        self.current = Status.STATUS_CONFIGURING
                        self.enter_configure.emit()
                    case Transition.TRANSITION_RESET:
                        self.current = Status.STATUS_RESETTING
                        self.enter_reset.emit()
                    case _:
                        self.current = Status.STATUS_FAILED
                        self.enter_failed.emit()

    @contextlib.contextmanager
    def managed_configure(self) -> Generator["State", None, None]:
        self.transition(Transition.TRANSITION_CONFIGURE)
        try:
            yield self
        finally:
            self.transition(Transition.TRANSITION_END_CONFIGURE)
