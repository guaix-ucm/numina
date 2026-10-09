#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Simulation of detectors"""

import numpy

from ..hwdevice import HWDevice
from ..simulation.efficiency import Efficiency


class VirtualDetector:
    """A readout channel of a detector.

    Parameters
    ----------
    base : slice or tuple of slices
        Region of the detector read by the channel.
    geom : tuple
        Regions of the output image: the trimmed data, the prescan
        columns, the overscan columns and the overscan rows.
    directfun : callable
        Function that transforms the region `base` to the orientation of
        the output image.
    readpars : object
        Readout parameters, with attributes 'gain', 'bias' and 'ron'.
    """

    def __init__(self, base, geom, directfun, readpars):

        self.base = base
        self.trim, self.pcol, self.ocol, self.orow = geom

        self.direcfun = directfun

        self.readpars = readpars

    def readout_in_buffer(self, elec, final):
        """Read the electrons `elec` of the channel into the image `final`.

        The trimmed region receives the electrons divided by the gain, and
        all the regions the bias and a Gaussian readout noise.
        """

        final[self.trim] = self.direcfun(elec[self.base])

        final[self.trim] = final[self.trim] / self.readpars.gain

        # We could use different RON and BIAS in each section
        for section in [self.trim, self.pcol, self.ocol, self.orow]:
            final[section] = self.readpars.bias + numpy.random.normal(final[section], self.readpars.ron)

        return final


class DetectorBase(HWDevice):
    """A simulated detector, that accumulates electrons and reads them out.

    Parameters
    ----------
    name : str
        Name of the device.
    shape : tuple of int
        Shape of the detector.
    qe : float, optional
        Quantum efficiency.
    qe_wl : optional
        Quantum efficiency as a function of the wavelength, an object
        with a method 'response(wl)', 1 for all wavelengths by default.
    dark : float, optional
        Dark current, in electrons per second.
    """

    def __init__(self, name, shape, qe=1.0, qe_wl=None, dark=0.0):

        super().__init__(name)

        self.dshape = shape
        self.pixscale = 15.0e-3

        self._det = numpy.zeros(shape, dtype="float64")

        self.qe = qe

        if qe_wl is None:
            # Efficiency 1 for all wavelengths
            self._qe_wl = Efficiency()
        else:
            self._qe_wl = qe_wl

        self.dark = dark
        # Exposure time since last reset
        self._time_last = 0.0

    def qe_wl(self, wl):
        """QE per wavelength."""
        return self._qe_wl.response(wl)

    def expose(self, source=0.0, time=0.0):
        """Accumulate the electrons of `source` and of the dark current during `time`"""
        self._time_last = time
        self._det += (self.qe * source + self.dark) * time

    def reset(self):
        """Reset the detector."""
        self._det[:] = 0.0

    def saturate(self, x):
        """Apply the saturation, nothing by default"""
        return x

    def simulate_poisson_variate(self):
        """Return a Poisson realization of the accumulated electrons"""
        elec_mean = self._det
        elec = numpy.random.poisson(elec_mean)
        return elec

    def pre_readout(self, elec_pre):
        """Process the electrons before the readout, nothing by default"""
        return elec_pre

    def base_readout(self, elec_f):
        """Convert the electrons to ADU, nothing by default"""
        return elec_f

    def post_readout(self, adu_r):
        """Clip the ADU to the range of uint16 and convert them"""
        adu_p = numpy.clip(adu_r, 0, 2**16 - 1)
        return adu_p.astype("uint16")

    def clean_up(self):
        """Prepare the detector after the readout, a reset by default"""
        self.reset()

    def readout(self):
        """Read out the detector and return the image in ADU.

        The steps are :meth:`simulate_poisson_variate`, :meth:`saturate`,
        :meth:`pre_readout`, :meth:`base_readout`, :meth:`post_readout`
        and :meth:`clean_up`, that subclasses can redefine.
        """

        elec = self.simulate_poisson_variate()

        elec_pre = self.saturate(elec)

        elec_f = self.pre_readout(elec_pre)

        adu_r = self.base_readout(elec_f)

        adu_p = self.post_readout(adu_r)

        self.clean_up()

        return adu_p
