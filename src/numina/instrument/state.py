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

    ACTIVE = 1
    FAILED = 2
    RESETTING = 3
    CONFIGURING = 4


class Transition(enum.Enum):
    RESET = 0
    END_RESET = 1
    CONFIGURE = 2
    END_CONFIGURE = 3
    ERROR = 4


class State:
    def __init__(self):
        self.current = Status.ACTIVE
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
        return self.current == Status.CONFIGURING

    def transition(self, inpt: Transition) -> None:
        match self.current:
            case Status.FAILED:
                match inpt:
                    case Transition.RESET:
                        self.exit_failed.emit()
                        self.current = Status.RESETTING
                        self.enter_reset.emit()
                    case _:
                        pass  # No action
            case Status.RESETTING:
                match inpt:
                    case Transition.RESET:
                        pass  # No action
                    case Transition.END_RESET:
                        self.exit_reset.emit()
                        self.current = Status.ACTIVE
                        self.enter_active.emit()
                    case _:
                        self.exit_reset.emit()
                        self.current = Status.FAILED
                        self.enter_failed.emit()
            case Status.CONFIGURING:
                match inpt:
                    case Transition.CONFIGURE:
                        pass  # No action
                    case Transition.END_CONFIGURE:
                        self.current = Status.ACTIVE
                        self.exit_configure.emit()
                    case _:
                        self.current = Status.FAILED
                        self.enter_failed.emit()
            case Status.ACTIVE:
                match inpt:
                    case Transition.CONFIGURE:
                        self.current = Status.CONFIGURING
                        self.enter_configure.emit()
                    case Transition.RESET:
                        self.current = Status.RESETTING
                        self.enter_reset.emit()
                    case _:
                        self.current = Status.FAILED
                        self.enter_failed.emit()

    @contextlib.contextmanager
    def managed_configure(self) -> Generator["State", None, None]:
        self.transition(Transition.CONFIGURE)
        try:
            yield self
        finally:
            self.transition(Transition.END_CONFIGURE)
