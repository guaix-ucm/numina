#
# Copyright 2015-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Function to retrieve tags from Observation results."""


def get_tags_from_full_ob(ob, reqtags=None):
    """
    Parameters
    ----------
    ob (ObservationResult): Observation result
    reqtags (dict): Keywords

    Returns
    -------
    A dictionary

    """
    # each instrument should have one
    # perhaps each mode...
    files = ob.frames
    cfiles = ob.children
    alltags = {}

    if reqtags is None:
        reqtags = []

    # Init alltags...
    # Open first image
    if files:
        for fname in files[:1]:
            with fname.open() as fd:
                header = fd[0].header
                for t in reqtags:
                    alltags[t] = header[t]
    else:

        for prod in cfiles[:1]:
            prodtags = prod.tags
            for t in reqtags:
                alltags[t] = prodtags[t]

    for fname in files:
        with fname.open() as fd:
            header = fd[0].header

            for t in reqtags:
                if alltags[t] != header[t]:
                    msg = f'wrong tag {t} in file {fname}'
                    raise ValueError(msg)

    for prod in cfiles:
        prodtags = prod.tags
        for t in reqtags:
            if alltags[t] != prodtags[t]:
                msg = f'wrong tag {t} in product {prod}'
                raise ValueError(msg)

    return alltags
