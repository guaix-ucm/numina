#
# Copyright 2008-2009 Sergio Pascual, Nicolas Cardiel
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

# $Id$

'''Recipe for the reduction of multiobject spectroscopy.

Recipe to reduce observations obtained in multiobject spectroscopy,
considering different possibilities depending on the size of the offsets along
the slits between individual exposures.  In particular, the following
strategies are considered: stare spectra, dithered/nodded spectra, and offset
spectra.

The images are proceed as in the Basic Reduction for Direct Imaging.  Spectra
are extracted (information about slit position is required), wavelength
calibrated and atmospheric extinction corrected.

**Inputs:**

**Outputs:**

**Procedure:**

'''
