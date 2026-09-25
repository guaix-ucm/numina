#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Auxiliary functions for initializing scripts with command line arguments and logging."""

from datetime import datetime
import logging
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from pathlib import Path

from numina._version import __version__
from numina.user.console import NuminaConsole


class SciNotationHighlighter(ReprHighlighter):
    """ReprHighlighter that also recognises uppercase exponents (e.g. 1.5E-13).

    The default ReprHighlighter only matches lowercase exponents, so numbers
    such as those found in FITS header cards are highlighted incorrectly.
    """

    highlights = ReprHighlighter.highlights + [
        r"(?P<number>(?<![\w.])[-+]?\d+\.?\d*[eE][-+]?\d+\b)",
    ]


def include_default_arguments_for_common_actions(
    parser,
    include_version=True,
    include_output_dir=True,
    include_no_color=True,
    include_record=True,
    include_echo=True,
    include_log_level=True,
):
    """Include default arguments to record and other common actions.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser object to which the default arguments will be added.
    include_version : bool, optional
        Whether to include the --version argument (default: True).
    include_output_dir : bool, optional
        Whether to include the --output-dir argument (default: True).
    include_record : bool, optional
        Whether to include the --record argument (default: True).
    include_echo : bool, optional
        Whether to include the --echo argument (default: True).
    include_log_level : bool, optional
        Whether to include the --log-level argument (default: True).
    """
    if include_version:
        parser.add_argument("--version", help="Display version information and exit", action="store_true")
    if include_output_dir:
        parser.add_argument("--output-dir", help="Output directory (default: .)", type=str, default=".")
    if include_no_color:
        parser.add_argument("--no-color", help="Disable color output", action="store_true")
    if include_record:
        parser.add_argument("--record", help="Record terminal output", action="store_true")
    if include_echo:
        parser.add_argument("--echo", help="Display full command line", action="store_true")
    if include_log_level:
        parser.add_argument(
            "--log-level",
            help="Set the logging level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
        )


def initialize_script_with_args(sys_argv, parser, args, local_name):
    """Initialize script with command line arguments and logging.

    This function initializes the script by parsing command line arguments,
    configuring logging, and displaying version information.
    It also provides a welcome message and handles the display of the
    full command line if requested. This functionality is useful for
    several scripts that require similar initialization steps.

    Parameters
    ----------
    sys_argv : list
        List of command line arguments (sys.argv).
    parser : argparse.ArgumentParser
        Argument parser object.
    args : argparse.Namespace
        Parsed command line arguments.
    local_name : str
        Name of the local script (usually __name__).

    Returns
    -------
    console : NuminaConsole
        Console object for rich output.
    logger : logging.Logger
        Logger object for logging messages.
    datetime_ini : datetime
        Start time of the script execution.
    """
    if len(sys_argv) == 1:
        parser.print_usage()
        raise SystemExit()

    # Initialize datetime for script execution
    datetime_ini = datetime.now()

    # Configure rich console
    highlighter = SciNotationHighlighter()
    no_color = getattr(args, "no_color", False)
    if no_color:
        console = NuminaConsole(record=args.record, color_system=None)
    else:
        console = NuminaConsole(record=args.record)
    console.highlighter = highlighter  # affect console.print()

    # Display version and exit if requested
    if hasattr(args, "version") and args.version:
        console.print(__version__)
        raise SystemExit()

    # Display full command line if requested
    if hasattr(args, "echo") and args.echo:
        console.print(f"[bright_red]Executing:\n{' '.join(sys_argv)}[/bright_red]\n", end="")

    # Configure logging
    if not hasattr(args, "log_level"):
        args.log_level = "INFO"

    handler_kwargs = dict(console=console, show_time=False, markup=True, highlighter=highlighter)
    if args.log_level in ["DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        format_log = "%(name)s %(levelname)s\n%(message)s"
    else:
        format_log = "%(message)s"
        handler_kwargs.update(show_path=False, show_level=False)
    handlers = [RichHandler(**handler_kwargs)]
    logging.basicConfig(level=args.log_level, format=format_log, handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib debug logs

    # Welcome message
    console.rule(f"[bold magenta]Welcome to {Path(sys_argv[0]).name}[/bold magenta]")

    # Display version info
    logger = logging.getLogger(local_name)
    logger.info(f"Using {local_name}")
    logger.info(f"Version {__version__}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Command line arguments: {args}")

    # Generate output directory if it does not exist
    if args.output_dir != ".":
        output_dir_path = Path(args.output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir_path}")

    return console, logger, datetime_ini


def goodbye_message_and_save_console(logger, console, datetime_ini, args_record, args_output_dir):
    """Display goodbye message and save console log if recording is enabled.

    Parameters
    ----------
    logger : logging.Logger
        Logger object for logging messages.
    console : NuminaConsole
        Console object for rich output.
    datetime_ini : datetime
        Start time of the script execution.
    args_record : bool
        Flag indicating whether to record the console output.
    args_output_dir : str
        Output directory where the console log will be saved if recording is enabled.
    """
    datatime_end = datetime.now()
    time_elapsed = datatime_end - datetime_ini
    logger.info("Total time elapsed: %s", str(time_elapsed))

    # Goodbye message
    console.rule("[bold magenta] Goodbye! [/bold magenta]")

    # Save console log if recording is enabled
    if args_record:
        output_dir_path = Path(args_output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = Path(args_output_dir) / "terminal_output.txt"
        with open(log_filename, "wt") as f:
            f.write(console.export_text(styles=True))
        logger.info(f"terminal output recorded in [green]{log_filename}[/green]")
