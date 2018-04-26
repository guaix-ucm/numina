#
# Copyright 2011-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DRP system wide initialization"""


from .drpsystem import DrpSystem

_system_drps = None


def get_system_drps():
    """Load all compatible DRPs in the system"""
    global _system_drps
    if _system_drps is None:
        _system_drps = DrpSystem()
        _system_drps.load()

    return _system_drps