#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy

from numina.instrument.hwdevice import HWDevice
from numina.instrument.simulation.efficiency import Efficiency


class VirtualDetector(object):
    """Each of the channels."""
    def __init__(self, base, geom, directfun, readpars):

        self.base = base
        self.trim, self.pcol, self.ocol, self.orow = geom

        self.direcfun = directfun

        self.readpars = readpars

    def readout_in_buffer(self, elec, final):

        final[self.trim] = self.direcfun(elec[self.base])

        final[self.trim] = final[self.trim] / self.readpars.gain

        # We could use different RON and BIAS in each section
        for section in [self.trim, self.pcol, self.ocol, self.orow]:
            final[section] = self.readpars.bias + numpy.random.normal(final[section], self.readpars.ron)

        return final


class DetectorBase(HWDevice):
    def __init__(self, name, shape, qe=1.0, qe_wl=None, dark=0.0):

        super(DetectorBase, self).__init__(name)

        self.dshape = shape
        self.pixscale = 15.0e-3

        self._det = numpy.zeros(shape, dtype='float64')

        self.qe = qe

        if qe_wl is None:
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
        self._time_last = time
        self._det += (self.qe * source + self.dark) * time

    def reset(self):
        """Reset the detector."""
        self._det[:] = 0.0

    def saturate(self, x):
        return x

    def simulate_poisson_variate(self):
        elec_mean = self._det
        elec = numpy.random.poisson(elec_mean)
        return elec

    def pre_readout(self, elec_pre):
        return elec_pre

    def base_readout(self, elec_f):
        return elec_f

    def post_readout(self, adu_r):
        adu_p = numpy.clip(adu_r, 0, 2**16-1)
        return adu_p.astype('uint16')

    def clean_up(self):
        self.reset()

    def readout(self):
        """Readout the detector."""

        elec = self.simulate_poisson_variate()

        elec_pre = self.saturate(elec)

        elec_f = self.pre_readout(elec_pre)

        adu_r = self.base_readout(elec_f)

        adu_p = self.post_readout(adu_r)

        self.clean_up()

        return adu_p
