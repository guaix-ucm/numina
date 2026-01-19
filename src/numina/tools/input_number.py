#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Input a number (integer or float) from the user with validation."""

from typing import Optional, Callable, Literal, Union

Number = Union[int, float]


def input_number(
    expected_type: Literal["int", "float"] = "int",
    min_val: Optional[Number] = None,
    max_val: Optional[Number] = None,
    prompt: Optional[str] = None,
    default: Optional[Number] = None,
    allow_blank_to_default: bool = True,
    attempts: Optional[int] = None,
    validator: Optional[Callable[[Number], bool]] = None,
    input_func: Callable[[str], str] = input,
) -> Number:
    """
    Prompt the user for a number (integer or float), with optional bounds and validation.

    Parameters
    ----------
    expected_type : {'int', 'float'}, default 'int'
        Numeric type to parse and return.
    min_val : int | float | None, default None
        Inclusive lower bound. If None, no lower bound is enforced.
    max_val : int | float | None, default None
        Inclusive upper bound. If None, no upper bound is enforced.
    prompt : str | None, default None
        Custom prompt text. If None, a context-aware prompt is generated.
    default : int | float | None, default None
        Default value returned when user presses Enter (if allow_blank_to_default=True).
        Must satisfy bounds and validator (if provided).
    allow_blank_to_default : bool, default True
        If True and default is not None, a blank line (Enter) returns the default.
    attempts : int | None, default None
        Maximum number of attempts before raising ValueError. If None, loop indefinitely.
    validator : callable(x) -> bool | None, default None
        Additional constraint; the parsed value must satisfy validator(x) == True.
    input_func : callable(prompt: str) -> str, default built-in input
        The function used to obtain input (useful for testing).

    Returns
    -------
    int | float
        The parsed and validated number.

    Raises
    ------
    TypeError
        If min_val, max_val, or default are provided with non-numeric types.
    ValueError
        If bounds are inconsistent, default violates constraints,
        expected_type is invalid, or attempts are exhausted.

    Notes
    -----
    - Bounds are **inclusive**.
    - For `expected_type='int'`, inputs like "3.0" are accepted **only** if they
      represent an exact integer value (no fractional part after conversion).

    Examples
    --------
    Basic (no bounds), default accepted by pressing Enter:

    >>> def fake_input_gen(vals):
    ...     it = iter(vals)
    ...     return lambda _: next(it)
    ...
    >>> inp = fake_input_gen([""])  # user presses Enter
    >>> input_number(expected_type="int", default=10, input_func=inp)
    10

    Integer with bounds [0, 8] and a provided value:

    >>> inp = fake_input_gen(["7"])
    >>> input_number("int", 0, 8, prompt="Pick [0-8]", input_func=inp)
    7

    Float in [0.0, 1.0], rejecting an out-of-range first try:

    >>> inp = fake_input_gen(["1.5", "0.75"])
    >>> input_number("float", 0.0, 1.0, prompt="Probability [0-1]", input_func=inp)
    0.75

    Integer parsing allows strings like "3.0" that are exactly integral:

    >>> inp = fake_input_gen(["3.0"])
    >>> input_number("int", input_func=inp)
    3

    But rejects non-integral numerals for integers:

    >>> inp = fake_input_gen(["3.14", "2"])
    >>> input_number("int", input_func=inp)
    Invalid input. Please enter a integer.
    2

    Custom validator: even non-negative integer:

    >>> is_even = lambda x: (x % 2 == 0)
    >>> inp = fake_input_gen(["-2", "3", "4"])
    >>> input_number("int", min_val=0, validator=is_even, prompt="Even ≥ 0", input_func=inp)
    Out of range or invalid. Enter a value ≥ 0.
    Out of range or invalid. Enter a value ≥ 0.
    4

    Attempts cap: stop after 2 invalid tries:

    >>> inp = fake_input_gen(["abc", "999"])
    >>> try:
    ...     _ = input_number("int", 0, 10, attempts=2, input_func=inp)
    ... except ValueError as e:
    ...     print("ERR:", str(e)[:34])  # shorten for doctest stability
    ERR: Maximum number of attempts reached

    Using only a minimum bound (no maximum):

    >>> inp = fake_input_gen(["-1", "0"])
    >>> input_number("int", min_val=0, input_func=inp)
    Out of range or invalid. Enter a value ≥ 0.
    0

    Using only a maximum bound (no minimum):

    >>> inp = fake_input_gen(["11", "10"])
    >>> input_number("int", max_val=10, input_func=inp)
    Out of range or invalid. Enter a value ≤ 10.
    10
    """
    # --- Validate expected_type ---
    if expected_type not in ("int", "float"):
        raise ValueError("expected_type must be 'int' or 'float'.")

    # --- Validate bounds consistency ---
    def _is_number(x) -> bool:
        return isinstance(x, (int, float))

    if min_val is not None and not _is_number(min_val):
        raise TypeError("min_val must be a number or None.")
    if max_val is not None and not _is_number(max_val):
        raise TypeError("max_val must be a number or None.")
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError("min_val must be <= max_val when both are provided.")

    # --- Validate default ---
    if default is not None and not _is_number(default):
        raise TypeError("default must be a number or None.")
    if default is not None:
        if min_val is not None and default < min_val:
            raise ValueError("Default is below the minimum bound.")
        if max_val is not None and default > max_val:
            raise ValueError("Default is above the maximum bound.")
        if validator is not None and not validator(default):
            raise ValueError("Default does not satisfy the validator constraint.")
        # Enforce type: if expected int, default must be an integer value
        if expected_type == "int" and not float(default).is_integer():
            raise ValueError("Default must be an integer for expected_type='int'.")

    # --- Build a context-aware prompt ---
    def _range_text() -> str:
        if min_val is not None and max_val is not None:
            return f"[{min_val}–{max_val}]"
        elif min_val is not None:
            return f"≥ {min_val}"
        elif max_val is not None:
            return f"≤ {max_val}"
        else:
            return ""

    type_label = "integer" if expected_type == "int" else "float"
    if prompt is None:
        base = f"Enter a {type_label}"
        rt = _range_text()
        if rt:
            base += f" {rt}"
    else:
        base = prompt

    default_str = f" [{default}]" if default is not None else None
    full_prompt = f"{base}{default_str}: " if (default is not None and allow_blank_to_default) else f"{base}: "

    # --- Parsing helper ---
    def _parse(s: str) -> Number:
        if expected_type == "int":
            # Accept inputs like "3.0" only if they represent an integer value exactly
            n_float = float(s)
            if not n_float.is_integer():
                raise ValueError("Expected an integer.")
            return int(n_float)
        else:
            return float(s)

    # --- Bounds & validator check ---
    def _valid(n: Number) -> bool:
        if min_val is not None and n < min_val:
            return False
        if max_val is not None and n > max_val:
            return False
        if validator is not None and not validator(n):
            return False
        return True

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
                return int(default) if expected_type == "int" else float(default)
            print("No input provided.", end="")
            if default is not None and not allow_blank_to_default:
                print(" (Blank does not accept default.)")
            else:
                print()
            if remaining is not None:
                remaining -= 1
            continue

        # Parse
        try:
            n = _parse(s)
        except ValueError:
            print(f"Invalid input. Please enter a {type_label}.")
            if remaining is not None:
                remaining -= 1
            continue

        # Check constraints
        if not _valid(n):
            if min_val is not None and max_val is not None:
                print(f"Out of range or invalid. Enter a value between {min_val} and {max_val}.")
            elif min_val is not None:
                print(f"Out of range or invalid. Enter a value ≥ {min_val}.")
            elif max_val is not None:
                print(f"Out of range or invalid. Enter a value ≤ {max_val}.")
            else:
                print("Invalid value for the provided constraints.")
            if remaining is not None:
                remaining -= 1
            continue

        return n
