#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


from __future__ import division
from __future__ import print_function

import os


def arg_file_is_new(parser, arg, mode='w'):
    """Auxiliary function to give an error if the file already exists.

    Parameters
    ----------
    parser : parser object
        Instance of argparse.ArgumentParser()
    arg : string
        File name.
    mode : string
        Optional string that specifies the mode in which the file is
        opened.

    Returns
    -------
    handler : file object
        Open file handle.

    """
    if os.path.exists(arg):
        parser.error("\nThe file \"%s\"\nalready exists and "
                     "cannot be overwritten!" % arg)
    else:
        # return an open file handle
        handler = open(arg, mode=mode)
        return handler
