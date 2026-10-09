#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Sequences base class"""


class Sequence:
    """Base class of the sequences of actions of an observing mode.

    Parameters
    ----------
    instrument : str
        Name of the instrument.
    mode : str
        Observing mode.
    """

    def __init__(self, instrument, mode):
        self.instrument = instrument
        self.mode = mode

    def setup_instrument(self, instrument):
        """Configure `instrument` for the sequence, nothing by default"""
        pass

    def run(self, **kwds):
        """Run the sequence, to be implemented by subclasses"""
        raise NotImplementedError
