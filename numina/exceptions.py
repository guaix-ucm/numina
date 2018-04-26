#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Exceptions for the numina package."""


import numina.ext.gtc


class Error(Exception):
    """Base class for exceptions in the numina package."""
    pass


class RecipeError(Error):
    """A non recoverable problem during recipe execution."""
    pass


class DetectorElapseError(Error):
    """Error in the clocking of a Detector."""
    pass


class DetectorReadoutError(Error):
    """Error in the readout of a Detector."""
    pass


class ValidationError(Exception):
    """Error during validation of Recipe inputs and outputs."""
    pass


class NoResultFound(Exception):
    """No result found in a DAL query."""
    pass

NoResultFoundOrig = NoResultFound


# If we are in the GCS environment, use its exceptions
# where applies
if numina.ext.gtc.check_gtc():
    import gtc.SSL.GCSTypes

    # Save original exception
    NoResultFoundOrig = NoResultFound
    NoResultFound = gtc.SSL.GCSTypes.NotFound
