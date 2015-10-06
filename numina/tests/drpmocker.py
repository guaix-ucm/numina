#
# Copyright 2015 Universidad Complutense de Madrid
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

"""A class to mock the DRP loading process."""

import pkg_resources

import numina.core.pipelineload as pload


def create_mock_entry_point(monkeypatch, entry_name, drpdata):

    loader = "%s.loader" % entry_name

    ep = pkg_resources.EntryPoint(entry_name, loader)

    def fake_loader():
        return pload.drp_load_data(drpdata)

    monkeypatch.setattr(ep, 'load', lambda: fake_loader)

    return ep


class DRPMocker(object):
    """Mocks the DRP loading process for testing."""
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self._eps = []

        def mockreturn(group=None):
            return self._eps

        self.monkeypatch.setattr(pkg_resources, 'iter_entry_points', mockreturn)

    def add_drp(self, name, data):
        ep = create_mock_entry_point(self.monkeypatch, name, data)
        self._eps.append(ep)
