#
# Copyright 2011-2014 Universidad Complutense de Madrid
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

"""DRP loader and related classes"""

import warnings
import pkg_resources


class Pipeline(object):
    """Base class for pipelines."""
    def __init__(self, name, recipes, version=1):
        self.name = name
        self.recipes = recipes
        self.version = version

    def get_recipe(self, mode):
        return self.recipes[mode]


class InstrumentConfiguration(object):
    """Configuration of an Instrument."""
    def __init__(self, values):
        self.values = values


class InstrumentDRP(object):
    """Description of an Instrument Data Reduction Pipeline"""
    def __init__(self, name, configurations, modes, pipelines, products=None):
        self.name = name
        self.configurations = configurations
        self.modes = modes
        self.pipelines = pipelines
        self.products = products
        if products is None:
            self.products = []


class ObservingMode(object):
    """Observing modes of an Instrument."""
    def __init__(self):
        self.name = ''
        self.uuid = ''
        self.key = ''
        self.url = ''
        self.instrument = ''
        self.summary = ''
        self.description = ''
        self.status = ''
        self.date = ''
        self.reference = ''
        self.tagger = None


class DrpSystem(object):
    """Load DRPs from the system."""

    ENTRY = 'numina.pipeline.1'

    def __init__(self):

        # Store queried DRPs
        self._drp_cache = {}

    def query_by_name(self, name):
        """Cached version of 'query_drp_system'"""
        if name in self._drp_cache:
            return self._drp_cache[name]
        else:
            drp = self._query_by_name(name)
            if drp:
                self._drp_cache[name] = drp
            return drp

    def _query_by_name(self, name):
        """Load a DRPs in 'numina.pipeline' entry_point by name"""

        for entry in pkg_resources.iter_entry_points(group=DrpSystem.ENTRY):
            if entry.name == name:
                drp_loader = entry.load()
                drpins = drp_loader()

                if self.instrumentdrp_check(drpins, entry.name):
                    return drpins
                else:
                    return None
        else:
            return None

    def query_all(self):
        """Return all available DRPs in 'numina.pipeline' entry_point."""

        drps = {}

        for entry in pkg_resources.iter_entry_points(group=DrpSystem.ENTRY):
            drp_loader = entry.load()
            drpins = drp_loader()
            if self.instrumentdrp_check(drpins, entry.name):
                drps[drpins.name] = drpins

        # Update cache
        self._drp_cache = drps

        return drps

    def instrumentdrp_check(self, drpins, entryname):
        if isinstance(drpins, InstrumentDRP):
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
