#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

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
        parser.error(f"\nThe file \"{arg}\"\nalready exists and cannot be overwritten!")
    else:
        # return an open file handle
        handler = open(arg, mode=mode)
        return handler
