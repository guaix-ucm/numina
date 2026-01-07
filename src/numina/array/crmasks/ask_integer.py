#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Prompt user for an integer values."""

from typing import Optional, Callable


def ask_integer(
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    prompt: Optional[str] = None,
    default: Optional[int] = None,
    allow_blank_to_default: bool = True,
    attempts: Optional[int] = None,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Prompt the user for an integer, optionally constrained by min_val/max_val.

    Parameters
    ----------
    min_val : int | None
        Minimum allowed integer (inclusive). If None, no lower bound is enforced.
    max_val : int | None
        Maximum allowed integer (inclusive). If None, no upper bound is enforced.
    prompt : str | None
        Custom prompt text. If None, a context-aware message is generated.
    default : int | None
        Default returned when user presses Enter (if allow_blank_to_default=True).
        Must satisfy provided bounds (if any).
    allow_blank_to_default : bool
        If True and default is not None, pressing Enter returns the default.
    attempts : int | None
        Max number of attempts. If None, keep prompting indefinitely.
        If the limit is reached without valid input, a ValueError is raised.
    input_func : callable
        Function used to read input. Defaults to built-in input (useful for testing).

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If min_val/max_val/default are not integers when provided.
    ValueError
        If min_val > max_val (when both provided), default violates bounds,
        or attempts are exhausted without valid input.
    """

    # --- Validate bounds types ---
    if min_val is not None and not isinstance(min_val, int):
        raise TypeError("min_val must be an integer or None.")
    if max_val is not None and not isinstance(max_val, int):
        raise TypeError("max_val must be an integer or None.")
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError("min_val must be <= max_val when both are provided.")

    # --- Validate default ---
    if default is not None and not isinstance(default, int):
        raise TypeError("default must be an integer or None.")
    if default is not None:
        if min_val is not None and default < min_val:
            raise ValueError("Default is below the minimum bound.")
        if max_val is not None and default > max_val:
            raise ValueError("Default is above the maximum bound.")

    # --- Compose a helpful prompt ---
    if prompt is None:
        # Build a range description depending on which bounds exist
        if min_val is not None and max_val is not None:
            base = f"Enter an integer [{min_val}–{max_val}]"
        elif min_val is not None:
            base = f"Enter an integer ≥ {min_val}"
        elif max_val is not None:
            base = f"Enter an integer ≤ {max_val}"
        else:
            base = "Enter an integer"
    else:
        base = prompt

    full_prompt = f"{base} (default={default}): " if (default is not None and allow_blank_to_default) else f"{base}: "

    # --- Prompt loop ---
    remaining = attempts
    while True:
        if remaining is not None and remaining <= 0:
            raise ValueError("Maximum number of attempts reached without valid input.")

        s = input_func(full_prompt)
        s = "" if s is None else str(s).strip()

        # Blank handling
        if s == "":
            if allow_blank_to_default and default is not None:
                return default
            print("No input provided.", end="")
            if default is not None and not allow_blank_to_default:
                print(" (Blank does not accept default.)")
            else:
                print()
            if remaining is not None:
                remaining -= 1
            continue

        # Parse integer
        try:
            n = int(s)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            if remaining is not None:
                remaining -= 1
            continue

        # Bounds check (only apply those that exist)
        if (min_val is not None and n < min_val) or (max_val is not None and n > max_val):
            if min_val is not None and max_val is not None:
                print(f"Out of range. Please enter a value between {min_val} and {max_val}.")
            elif min_val is not None:
                print(f"Out of range. Please enter a value ≥ {min_val}.")
            else:  # max_val is not None
                print(f"Out of range. Please enter a value ≤ {max_val}.")
            if remaining is not None:
                remaining -= 1
            continue

        return n
