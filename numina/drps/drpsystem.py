#
# Copyright 2011-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""DRP system-wide loader"""

from __future__ import print_function

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
            raise KeyError('{}'.format(name))

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
