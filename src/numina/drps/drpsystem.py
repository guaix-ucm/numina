#
# Copyright 2011-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DRP system-wide loader"""

import sys
import pkg_resources

from .drpbase import DrpGeneric


class DrpSystem(DrpGeneric):
    """Load DRPs from the system."""

    def __init__(self, entry_point='numina.pipeline.1'):
        self.entry = entry_point
        super(DrpSystem, self).__init__()

    def load(self):
        """Load all available DRPs in 'entry_point'."""

        for drpins in self.iload(self.entry):
            self.drps[drpins.name] = drpins

        return self

    @classmethod
    def load_drp(self, name, entry_point='numina.pipeline.1'):
        """Load all available DRPs in 'entry_point'."""

        for drpins in self.iload(entry_point):
            if drpins.name == name:
                return drpins
        else:
            raise KeyError(f'{name}')

    @classmethod
    def iload(cls, entry_point='numina.pipeline.1'):
        """Load all available DRPs in 'entry_point'."""

        for entry in pkg_resources.iter_entry_points(group=entry_point):
            try:
                drp_loader = entry.load()
                drpins = drp_loader()
                if cls.instrumentdrp_check(drpins, entry.name):
                    yield drpins
            except Exception as error:
                print('Problem loading', entry, file=sys.stderr)
                print("Error is: ", error, file=sys.stderr)
