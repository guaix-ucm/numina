#
# Copyright 2011-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DRP storage class"""

import warnings


class DrpBase:
    """Store DRPs, base class"""

    def __init__(self):
        pass

    def query_by_name(self, name):
        """Return the DRP named `name`, this base class has no DRPs and raises ``KeyError``."""
        raise KeyError(name)

    def query_all(self):
        """Return the DRPs by name, an empty dictionary in this base class."""
        return {}

    @staticmethod
    def instrumentdrp_check(drpins, entryname):
        """Check whether `drpins` is an ``InstrumentDRP`` named `entryname`, warn with ``RuntimeWarning`` if not."""
        import numina.core.pipeline

        if isinstance(drpins, numina.core.pipeline.InstrumentDRP):
            if drpins.name == entryname:
                return True
            else:
                msg = f'Entry name "{entryname}" and DRP name "{drpins.name}" differ'
                warnings.warn(msg, RuntimeWarning)
                return False
        else:
            msg = f"Object {drpins!r} does not contain a valid DRP"
            warnings.warn(msg, RuntimeWarning)
            return False


class DrpGeneric(DrpBase):
    """Store DRPs.in a dictionary"""

    def __init__(self, drps=None):
        """Store `drps`, a dictionary of DRPs by name, empty if None."""
        super().__init__()
        self.drps = drps

    def query_by_name(self, name):
        """Query DRPs in internal storage by name"""
        return self._drps[name]

    def query_all(self):
        """Return all available DRPs in internal storage"""

        return self._drps

    @property
    def drps(self):
        """Dictionary of the DRPs, by name. Setting it to None stores an empty dictionary."""
        return self._drps

    @drps.setter
    def drps(self, newdrps):
        self._drps = {} if newdrps is None else newdrps
