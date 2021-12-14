#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""A class to mock the DRP loading process."""

import pkg_resources

import numina.core.pipelineload as pload


def create_mock_entry_point(monkeypatch, entry_name, drploader):

    loader = f"{entry_name}.loader"

    ep = pkg_resources.EntryPoint(entry_name, loader)

    monkeypatch.setattr(ep, 'load', lambda: drploader)

    return ep


class DRPMocker(object):
    """Mocks the DRP loading process for testing."""
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self._eps = []

        basevalue = pkg_resources.iter_entry_points
        # Use the mocker only for 'numina.pipeline.1'
        def mockreturn(group, name=None):
            if group == 'numina.pipeline.1':
                return self._eps
            else:
                return basevalue(group=group, name=name)

        self.monkeypatch.setattr(pkg_resources, 'iter_entry_points', mockreturn)

    def add_drp(self, name, loader):

        if callable(loader):
            ep = create_mock_entry_point(self.monkeypatch, name, loader)
        else:
            # Assume loader is data instead
            drpdata = loader

            def drploader():
                return pload.drp_load_data('numina', drpdata)

            ep = create_mock_entry_point(self.monkeypatch, name, drploader)

        self._eps.append(ep)
