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
from rich.markup import escape
import sys
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


class NuminaScriptDefinition:
    """Class to define a Numina script with command line arguments and logging.

    This class encapsulates the definition of a Numina script, including its
    argument parser, console, logger, and execution time. It provides methods
    to initialize the script with command line arguments and to display a
    goodbye message while saving the console log if recording is enabled.
    """

    def __init__(
        self,
        parser,
        argv=None,
        include_version=True,
        include_output_dir=True,
        include_no_color=True,
        include_record=True,
        include_echo=True,
        include_log_level=True,
    ):
        """Initialize the NuminaScriptDefinition with the provided argument parser and options.

        Parser and options are used to configure the command line arguments,
        console, and logging for the script.

        This method displays version information and a welcome message,
        and handles the display of the full command line if requested.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser object.
        argv : list of str or None, optional
            Command line arguments, excluding the program name.
            If None, sys.argv[1:] is used.
        include_version : bool, optional
            Whether to include the --version argument (default: True).
        include_output_dir : bool, optional
            Whether to include the --output-dir argument (default: True).
        include_no_color : bool, optional
            Whether to include the --no-color argument (default: True).
        include_record : bool, optional
            Whether to include the --record argument (default: True).
        include_echo : bool, optional
            Whether to include the --echo argument (default: True).
        include_log_level : bool, optional
            Whether to include the --log-level argument (default: True).
        """
        self.parser = parser
        self.args = None
        self.console = None
        self.logger = None
        self.datetime_ini = None

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

        if argv is None:
            argv = sys.argv[1:]

        if len(argv) == 0:
            parser.print_usage()
            raise SystemExit()

        self.args = parser.parse_args(argv)

        # Name of the module that called this function
        caller_name = sys._getframe(1).f_globals.get("__name__", "__main__")

        # Initialize datetime for script execution
        self.datetime_ini = datetime.now()

        # Configure rich console
        highlighter = SciNotationHighlighter()
        record = getattr(self.args, "record", False)
        if getattr(self.args, "no_color", False):
            self.console = NuminaConsole(record=record, color_system=None)
        else:
            self.console = NuminaConsole(record=record)
        self.console.highlighter = highlighter  # affects console.print()

        # Display version and exit if requested
        if getattr(self.args, "version", False):
            self.console.print(__version__)
            raise SystemExit()

        # Display full command line if requested
        if getattr(self.args, "echo", False):
            full_command = " ".join([parser.prog] + argv)
            self.console.print(f"[bright_red]Executing:\n{escape(full_command)}[/bright_red]\n", end="")

        # Configure logging
        log_level = getattr(self.args, "log_level", "INFO")
        handler_kwargs = dict(console=self.console, show_time=False, markup=True, highlighter=highlighter)
        if log_level == "INFO":
            format_log = "%(message)s"
            handler_kwargs.update(show_path=False, show_level=False)
        else:
            format_log = "%(name)s %(levelname)s\n%(message)s"
        logging.basicConfig(level=log_level, format=format_log, handlers=[RichHandler(**handler_kwargs)])
        logging.getLogger("matplotlib").setLevel(logging.ERROR)  # suppress matplotlib debug logs

        # Get the logger for the calling module
        self.logger = logging.getLogger(caller_name)

        # Welcome message and version info
        if self.logger.isEnabledFor(logging.INFO):
            self.console.rule(f"[bold magenta]Welcome to {escape(parser.prog)}[/bold magenta]")
            self.logger.info(f"Using {caller_name}")
            self.logger.info(f"Version {__version__}")

        self.logger.debug(f"Command line arguments: {self.args}", extra={"markup": False})

        # Generate output directory if it does not exist
        output_dir = getattr(self.args, "output_dir", ".")
        if output_dir != ".":
            output_dir_path = Path(output_dir)
            if not output_dir_path.exists():
                output_dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created output directory: {output_dir_path}")

    def goodbye_message_and_save_console(self):
        """Display goodbye message and save console log if recording is enabled."""
        # Get the current logging level
        current_logging_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())

        # Calculate the time elapsed
        time_elapsed = datetime.now() - self.datetime_ini
        if current_logging_level in ["NOTSET", "DEBUG", "INFO"]:
            self.logger.info("Total time elapsed: %s", str(time_elapsed))

        # Goodbye message
        if current_logging_level in ["NOTSET", "DEBUG", "INFO"]:
            self.console.rule("[bold magenta] Goodbye! [/bold magenta]")

        # Save console log if recording is enabled
        if self.args.record:
            if self.args.output_dir is None:
                self.logger.warning("Output directory is not properly specified. Terminal output will not be recorded.")
                return
            output_dir_path = Path(self.args.output_dir)
            if not output_dir_path.exists():
                output_dir_path.mkdir(parents=True, exist_ok=True)
            log_filename = Path(output_dir_path) / "terminal_output.txt"
            with open(log_filename, "wt") as f:
                f.write(self.console.export_text(styles=True))
            if current_logging_level in ["NOTSET", "DEBUG", "INFO"]:
                self.logger.info(f"Terminal output recorded in [green]{log_filename}[/green]")
