#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Skeleton script for Numina tools."""

import argparse
import logging
from rich_argparse import RichHelpFormatter
import sys

from .initialize_script_with_args import include_default_arguments_for_common_actions
from .initialize_script_with_args import initialize_script_with_args
from .initialize_script_with_args import goodbye_message_and_save_console


def auxiliary_function():
    """Auxiliary function for demonstration purposes."""
    logger = logging.getLogger(__name__)
    logger.info("This is an auxiliary function that can be used in the script.")


def main(args=None):
    """Main function"""

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="This is a skeleton script for Numina tools.", formatter_class=RichHelpFormatter
    )
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger, datetime_ini = initialize_script_with_args(sys.argv, parser, args, __name__)

    # Call the auxiliary function
    auxiliary_function()

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":
    main()
