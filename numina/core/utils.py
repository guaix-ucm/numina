#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Recipes for system checks. """

import logging

from numina.core import BaseRecipe, DataFrameType
from numina.core.requirements import ObservationResultRequirement
import numina.core.dataholders as dh
import numina.core.query as qry


_logger = logging.getLogger(__name__)


class AlwaysFailRecipe(BaseRecipe):
    """A Recipe that always fails."""

    def __init__(self, *args, **kwargs):
        super(AlwaysFailRecipe, self).__init__(
            version="1"
        )

    def run(self, requirements):
        raise TypeError('This Recipe always fails')


class AlwaysSuccessRecipe(BaseRecipe):
    """A Recipe that always successes."""

    def __init__(self, *args, **kwargs):
        super(AlwaysSuccessRecipe, self).__init__(
            version=1
        )

    def run(self, recipe_input):
        return self.create_result()


class OBSuccessRecipe(BaseRecipe):
    """A Recipe that always successes, it requires an OB"""

    obresult = ObservationResultRequirement()

    def __init__(self, *args, **kwargs):
        super(OBSuccessRecipe, self).__init__(
            version=1
        )

    def run(self, recipe_input):
        return self.create_result()


class Combine(BaseRecipe):

    obresult = ObservationResultRequirement()
    method = dh.Parameter('mean', "Method of combination")
    field = dh.Parameter('image', "Extract field of previous result")
    result = dh.Result(DataFrameType)

    def run(self, recipe_input):
        import numina.array.combine as c

        method = getattr(c, recipe_input.method)
        obresult = recipe_input.obresult
        result = combine_frames(obresult.frames, method)
        return self.create_result(result=result)

    def build_recipe_input(self, obsres, dal):
        import numina.exceptions

        result = {}
        result['obresult'] = obsres
        for key in ['method', 'field']:
            req = self.requirements()[key]
            query_option = self.query_options.get(key)
            try:
                result[key] = req.query(dal, obsres, options=query_option)
            except numina.exceptions.NoResultFound as notfound:
                req.on_query_not_found(notfound)
        if 'field' in result:
            # extract values:
            obreq = self.requirements()['obresult']
            qoptions = qry.ResultOf(field=result['field'])

            obsres = obreq.type.query("obresult", dal, obsres, options=qoptions)
            result['obresult'] = obsres

        rinput = self.create_input(**result)
        return rinput


def combine_frames(frames, method, errors=True, prolog=None):
    import astropy.io.fits as fits
    import datetime
    import uuid

    import numina.datamodel

    odata = []
    cdata = []
    datamodel = numina.datamodel.DataModel()
    try:
        _logger.info('processing input images')
        for frame in frames:
            hdulist = frame.open()
            fname = datamodel.get_imgid(hdulist)
            _logger.info('input is %s', fname)
            final = hdulist
            _logger.debug('output is input: %s', final is hdulist)
            cdata.append(final)
            # Files to be closed at the end
            odata.append(hdulist)
            if final is not hdulist:
                odata.append(final)

        base_header = cdata[0][0].header.copy()
        cnum = len(cdata)
        _logger.info("stacking %d images using '%s'", cnum, method.__name__)
        data = method([d[0].data for d in cdata], dtype='float32')
        hdu = fits.PrimaryHDU(data[0], header=base_header)
        _logger.debug('update result header')
        if prolog:
            _logger.debug('write prolog')
            hdu.header['history'] = prolog
        hdu.header['history'] = "Combined %d images using '%s'" % (cnum, method.__name__)
        hdu.header['history'] = 'Combination time {}'.format(datetime.datetime.utcnow().isoformat())
        for img in cdata:
            hdu.header['history'] = "Image {}".format(datamodel.get_imgid(img))
        prevnum = base_header.get('NUM-NCOM', 1)
        hdu.header['NUM-NCOM'] = prevnum * cnum
        hdu.header['UUID'] = str(uuid.uuid1())
        # Headers of last image
        hdu.header['TSUTC2'] = cdata[-1][0].header['TSUTC2']
        if errors:
            varhdu = fits.ImageHDU(data[1], name='VARIANCE')
            num = fits.ImageHDU(data[2], name='MAP')
            result = fits.HDUList([hdu, varhdu, num])
        else:
            result = fits.HDUList([hdu])
    finally:
        _logger.debug('closing images')
        for hdulist in odata:
            hdulist.close()

    return result
