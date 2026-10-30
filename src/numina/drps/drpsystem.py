#
# Copyright 2011-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DRP system-wide loader"""

from importlib.metadata import entry_points
import logging


from .drpbase import DrpGeneric

_logger = logging.getLogger(__name__)


class DrpSystem(DrpGeneric):
    """Load DRPs from the system."""

    def __init__(self, entry_point="numina.pipeline.1"):
        """Find the DRPs in the entry point group `entry_point`, they are not loaded until ``load`` is called."""
        self.entry = entry_point
        super().__init__()

    def load(self):
        """Load all available DRPs in 'entry_point'."""
        for drpins in self.iload(self.entry):
            self.drps[drpins.name] = drpins

        return self

    @classmethod
    def load_drp(cls, name, entry_point="numina.pipeline.1"):
        """Load all available DRPs in 'entry_point'."""

        for drpins in cls.iload(entry_point):
            if drpins.name == name:
                return drpins
        else:
            raise KeyError(f"{name}")

    @classmethod
    def iload(cls, entry_point="numina.pipeline.1"):
        """Load all available DRPs in 'entry_point'."""

        for entry in entry_points(group=entry_point):
            try:
                drp_loader = entry.load()
                drpins = drp_loader()
                if cls.instrumentdrp_check(drpins, entry.name):
                    yield drpins
            except Exception as error:
                _logger.warning("problem loading the DRP of %s: %s", entry, error)
