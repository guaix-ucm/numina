#
# Copyright 2017 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import inspect
from numina.exceptions import NoResultFound


class MultiType(object):
    def __init__(self, *args):
        self.options = []
        self.current = None
        for obj in args:
            if inspect.isclass(obj):
                obj = obj()
            self.options.append(obj)

    def isproduct(self):
        # FIXME: this is only need for docs, remove it
        return True

    def query(self, name, dal, ob, options=None):

        # Results for subtypes
        results = []
        for subtype in self.options:
            try:
                result = subtype.query(name, dal, ob, options=options)
                results.append((subtype, result))
            except NoResultFound as notfound:
                subtype.on_query_not_found(notfound)
        # Select wich of the results we choose
        if results:
            # Select the first, for the moment
            subtype, result = results[0]
            self.current = subtype
            return result
        else:
            raise NoResultFound

    def on_query_not_found(self, notfound):
        pass

    def convert_in(self, obj):
        if self.current:
            return self.current.convert_in(obj)

        raise ValueError('No query performed, current is None')
