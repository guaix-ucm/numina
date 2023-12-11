/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#ifndef NU_COMBINE_METHODS_H
#define NU_COMBINE_METHODS_H

#include "nu_combine_defs.h"

int NU_mean_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data);

int NU_sum_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data);

int NU_median_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data);

int NU_minmax_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data);

int NU_sigmaclip_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data);

int NU_quantileclip_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data);

void NU_destructor_function(void* cobject, void *cdata);

#endif // NU_COMBINE_METHODS_H
