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
    """Return the tags `tag_keys` of the observation result `obsres`.

    The tags are read from its first frame (see ``get_sample_frame``) with
    ``extract_tags_from_img``, the labels of `obsres` have precedence. Without
    frames, the tags are an empty dict.

    Raises
    ------
    ValueError
        With `strict`, if the tags of the frames or the results of `obsres`
        are different.
    """
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
        # obsres.children holds the ids of the children OBs. Their results,
        # when used, have been added to obsres.frames or obsres.results,
        # and have been checked in the previous loop
        return final_tags
    else:
        return final_tags


def extract_tags_from_img(img, tag_keys, datamodel: DataModel, base=None) -> dict:
    """Return the tags `tag_keys` of the image `img`.

    The values in the dict `base` have precedence, the others are extracted
    from `img` with the FITS extractor of `datamodel`.
    """

    base = base or {}
    fits_extractor = datamodel.extractor_map["fits"]
    final_tags = {}
    for key in tag_keys:

        if key in base:
            final_tags[key] = base[key]
        else:
            final_tags[key] = fits_extractor.extract(key, img)
    return final_tags
