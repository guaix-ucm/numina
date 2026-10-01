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

from .initialize_script_with_args import NuminaScriptDefinition


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
    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser)
    args = myscript.args  # noqa: F841
    logger = myscript.logger  # noqa: F841
    console = myscript.console  # noqa: F841

    # Call the auxiliary function
    auxiliary_function()

    # Display goodbye message and save console log if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
