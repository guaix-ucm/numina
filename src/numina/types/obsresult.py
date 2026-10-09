# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


from numina.core.oresult import ObservationResult

# import numina.core.instrument.insconf as insconf

from .frame import DataFrameType
from .datatype import DataType


def _obtain_mode(instrument, mode_key):
    """The observing mode `mode_key` of the DRP of `instrument`"""
    import numina.drps

    drps = numina.drps.get_system_drps()
    drp = drps.query_by_name(instrument)
    return drp.modes[mode_key]


def validate_raw_frames(obj, mode):
    """Check each frame of the observation result as a raw image of the mode.

    The images are checked with the function that the DRP of the instrument
    registers in :data:`numina.core.config.check`, as ``numina verify``
    does, with ``astype`` the raw image type of the mode (``rawimage`` in
    drp.yaml). If the DRP registers no function, or the mode has no raw
    image type, the images are not checked.

    Returns
    -------
    list of str
        The errors, one for each invalid image.
    """
    import numina.core.config as cfg

    rawimage = getattr(mode, "rawimage", None)
    if rawimage is None or obj.instrument not in cfg.check:
        return []

    errors = []
    for idx, frame in enumerate(obj.frames):
        name = getattr(frame, "filename", None) or f"frame {idx}"
        try:
            with frame.open() as hdulist:
                cfg.check(obj.instrument, hdulist, astype=rawimage)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}")
    return errors


class ObservationResultType(DataType):
    """The type of ObservationResult."""

    def __init__(self, rawtype=None):
        super().__init__(ptype=ObservationResult)
        if rawtype:
            self.rawtype = rawtype
        else:
            self.rawtype = DataFrameType

    def validate(self, obj):
        """Validate the observation result.

        Each frame is checked as a raw image of the observing mode (see
        :func:`validate_raw_frames`), and then the observation result with
        the validator of the mode (``validator`` in drp.yaml).

        Raises
        ------
        numina.exceptions.ValidationError
            If a frame is not a valid raw image of the mode.
        """
        import numina.exceptions

        mode = _obtain_mode(obj.instrument, obj.mode)
        errors = validate_raw_frames(obj, mode)
        if errors:
            raise numina.exceptions.ValidationError("invalid raw images: " + "; ".join(errors))
        validator = mode.validator or (lambda mod, obj: True)
        return validator(self, obj)
