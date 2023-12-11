#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

#cython: language_level=3

ctypedef fused datacube_t:
    double[:,:,:]
    float[:,:,:]
    long[:,:,:]
    int[:,:,:]

ctypedef fused result_t:
    double[:,:]
    float[:,:]

ctypedef char[:,:] mask_t

MASK_GOOD = 0
MASK_SATURATION = 3

