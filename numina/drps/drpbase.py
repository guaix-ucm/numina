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

"""DRP storage class"""

import warnings


class DrpBase(object):
    """Store DRPs, base class"""

    def __init__(self):
        pass

    def query_by_name(self, name):
        return None

    def query_all(self):
        return {}

    @staticmethod
    def instrumentdrp_check(drpins, entryname):
        import numina.core.pipeline
        if isinstance(drpins, numina.core.pipeline.InstrumentDRP):
            if drpins.name == entryname:
                return True
            else:
                msg = 'Entry name "{}" and DRP name "{}" differ'.format(entryname, drpins.name)
                warnings.warn(msg, RuntimeWarning)
                return False
        else:
            msg = 'Object {0!r} does not contain a valid DRP'.format(drpins)
            warnings.warn(msg, RuntimeWarning)
            return False


class DrpGeneric(DrpBase):
    """Store DRPs.in a dictionary"""

    def __init__(self, drps=None):
        super(DrpGeneric, self).__init__()
        self.drps = drps

    def query_by_name(self, name):
        """Query DRPs in internal storage by name

        """
        return self._drps[name]

    def query_all(self):
        """Return all available DRPs in internal storage"""

        return self._drps

    @property
    def drps(self):
        return self._drps

    @drps.setter
    def drps(self, newdrps):
        self._drps = {} if newdrps is None else newdrps