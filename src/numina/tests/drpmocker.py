#
# Copyright 2015-2023 Universidad Complutense de Madrid
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
        basevalue = importlib.metadata.entry_points
        # Use the mocker only for 'numina.pipeline.1'

        def mockreturn(group, name=None):
            if group == "numina.pipeline.1":
                return self._eps
            elif name is None:
                return basevalue(group=group)
            else:
                return basevalue(group=group, name=name)

        self.monkeypatch.setattr(numina.drps.drpsystem, "entry_points", mockreturn)

    def add_drp(self, name, loader):

        if callable(loader):
            ep = create_mock_entry_point(loader, name)
        else:
            # Assume loader is data instead
            drpdata = loader

            def drploader():
                return pload.drp_load_data("numina", drpdata)

            ep = create_mock_entry_point(drploader, name)

        self._eps.append(ep)
