#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Function to retrieve tags from Observation results."""

import itertools

from numina.datamodel import DataModel


def extract_tags_from_obsres(obsres, tag_keys, datamodel: DataModel, strict=True) -> dict:
    sample = obsres.get_sample_frame()
    if sample is None:
        return {}
    with sample.open() as ref_img:
        final_tags = extract_tags_from_img(ref_img, tag_keys, datamodel, base=obsres.labels)
    if strict:
        for frame in itertools.chain(obsres.frames, obsres.results.values()):
            with frame.open() as img:
                this_tags = extract_tags_from_img(img, tag_keys, datamodel, base=obsres.labels)
                if this_tags != final_tags:
                    raise ValueError(f"tags in image {frame} are {this_tags} ! = {final_tags}")

        for res in obsres.children:
            res_tags = res.tags
            for t in tag_keys:
                if final_tags[t] != res_tags[t]:
                    msg = f"wrong tag {t} in product {res}"
                    raise ValueError(msg)

        return final_tags
    else:
        return final_tags


def extract_tags_from_img(img, tag_keys, datamodel: DataModel, base=None) -> dict:

    base = base or {}
    fits_extractor = datamodel.extractor_map["fits"]
    final_tags = {}
    for key in tag_keys:

        if key in base:
            final_tags[key] = base[key]
        else:
            final_tags[key] = fits_extractor.extract(key, img)
    return final_tags
