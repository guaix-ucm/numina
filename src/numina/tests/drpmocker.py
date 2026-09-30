#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""A class to mock the DRP loading process."""

import importlib.metadata

import numina.core.pipelineload as pload
import numina.drps.drpsystem


def create_mock_entry_point(drploader, entry_name, group="numina.pipeline.1"):

    value = f"{entry_name}.loader"

    class EntryPoint(importlib.metadata.EntryPoint):
        def load(self):
            return drploader

    ep = EntryPoint(name=entry_name, value=value, group=group)
    return ep


class DRPMocker:
    """Mocks the DRP loading process for testing."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self._eps = []
        # Empty the cache of get_system_drps, so that
        # the DRPs are loaded again, with the mocked entry points
        self.monkeypatch.setattr(numina.drps, "_system_drps", None)
        basevalue = importlib.metadata.entry_points
        # Use the mocker only for 'numina.pipeline.1'

        def mock_return(group, name=None):
            if group == "numina.pipeline.1":
                return self._eps
            elif name is None:
                return basevalue(group=group)
            else:
                return basevalue(group=group, name=name)

        self.monkeypatch.setattr(numina.drps.drpsystem, "entry_points", mock_return)

    def add_drp(self, name, loader):

        if callable(loader):
            ep = create_mock_entry_point(loader, name)
        else:
            # Assume loader is data instead
            drp_data = loader

            def drp_loader():
                return pload.drp_load_data("numina", drp_data)

            ep = create_mock_entry_point(drp_loader, name)

        self._eps.append(ep)
