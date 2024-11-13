#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Function to retrieve tags from Observation results."""


import itertools
import warnings

from numina.datamodel import DataModel


def get_tags_from_full_ob(ob, reqtags=None):
    """
    Parameters
    ----------
    ob (ObservationResult): Observation result
    reqtags (iterable): Keywords

    Returns
    -------
    A dictionary

    """
    warnings.warn("'get_tags_from_full_ob' is deprecated, use 'extract_tags_from_obsres' instead",
                  DeprecationWarning, stacklevel=2)


    if reqtags is None:
        reqtags = []
    # adding an override from reqtags
    mappings = {r: r for r in reqtags}
    datamodel = DataModel(mappings=mappings)

    return extract_tags_from_obsres(ob, reqtags, datamodel, strict=True)


def extract_tags_from_obsres(obsres, tag_keys, datamodel: DataModel, strict=True) -> dict:
    sample = obsres.get_sample_frame()
    if sample is None:
        return {}
    ref_img = sample.open()
    final_tags = extract_tags_from_img(
        ref_img, tag_keys, datamodel, base=obsres.labels)
    if strict:
        for frame in itertools.chain(obsres.frames, obsres.results.values()):
            this_tags = extract_tags_from_img(
                frame.open(), tag_keys, datamodel, base=obsres.labels)
            print('this_tags', this_tags)
            if this_tags != final_tags:
                raise ValueError(f"tags in image {frame} are {this_tags} ! = {final_tags}")

        for res in obsres.children:
            res_tags = res.tags
            for t in tag_keys:
                if final_tags[t] != res_tags[t]:
                    msg = f'wrong tag {t} in product {res}'
                    raise ValueError(msg)

        return final_tags
    else:
        return final_tags


def extract_tags_from_img(img, tag_keys, datamodel: DataModel, base=None) -> dict:

    base = base or {}
    fits_extractor = datamodel.extractor_map['fits']
    final_tags = {}
    for key in tag_keys:

        if key in base:
            final_tags[key] = base[key]
        else:
            final_tags[key] = fits_extractor.extract(key, img)
    return final_tags
