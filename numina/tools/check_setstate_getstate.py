#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Double check that __setstate__ and __getstate__ methods work properly."""

import importlib


def check_setstate_getstate(self, json_file_name):
    class_name = self.__class__.__name__

    ClassName = getattr(importlib.import_module('emirdrp.products'),
                        class_name)

    # concatenate __getstate__ and __setstate__
    instance = ClassName()
    instance.__setstate__(self.__getstate__())
    instance.writeto(json_file_name + '_bis')

    # load data from input JSON file and save them again
    instance_bis = ClassName._datatype_load(json_file_name)
    instance_bis.writeto(json_file_name + '_bis2')
