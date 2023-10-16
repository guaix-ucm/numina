#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""A class to mock the DRP loading process."""

import importlib.metadata
import sys

import backports.entry_points_selectable
import numina.core.pipelineload as pload


def create_mock_entry_point(monkeypatch, entry_name, drploader):

    value = f"{entry_name}.loader"
    group = "numina.pipeline.1"

    # In python >= 3.11 EntryPoint is inmutable
    # and we cannot use monkeypatch
    # We remove __setattr__ that raises an exception
    if not sys.version_info < (3, 11):
        try:
            delattr(importlib.metadata.EntryPoint, '__setattr__')
        except AttributeError:
            pass

    ep = importlib.metadata.EntryPoint(
        name=entry_name, value=value, group=group)

    monkeypatch.setattr(ep, 'load', lambda: drploader)

    return ep


class DRPMocker(object):
    """Mocks the DRP loading process for testing."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self._eps = []
        basevalue = backports.entry_points_selectable.entry_points
        # Use the mocker only for 'numina.pipeline.1'

        def mockreturn(group, name=None):
            if group == 'numina.pipeline.1':
                return self._eps
            else:
                return basevalue(group=group, name=name)

        self.monkeypatch.setattr(
            backports.entry_points_selectable, 'entry_points', mockreturn)

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
