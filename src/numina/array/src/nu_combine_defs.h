/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#ifndef NU_COMBINE_DEFS_H
#define NU_COMBINE_DEFS_H

#define NU_COMBINE_OUTDIM 3

#include <cstddef>

typedef int (*CombineFunc)(double*, double*, size_t, double*[NU_COMBINE_OUTDIM], void*);

#endif // NU_COMBINE_DEFS_H
