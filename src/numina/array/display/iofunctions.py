#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import sys


def readc(prompt, default=None, valid=None, question_mark=True):
    """Return a single character read from keyboard

    Parameters
    ----------
    prompt : str
        Prompt string.
    default : str
        Default value.
    valid : str
        String providing valid characters. If None, all characters are
        valid (default).
    question_mark : bool
        If True, display question mark after prompt.

    Returns
    -------
    cresult : str
        Read value.

    """

    cresult = None  # Avoid PyCharm warning

    # question mark
    if question_mark:
        cquestion_mark = ' ? '
    else:
        cquestion_mark = ''

    # main loop
    loop = True
    while loop:

        # display prompt
        if default is None:
            print(prompt + cquestion_mark, end='')
            sys.stdout.flush()
        else:
            print(prompt + ' [' + str(default) + ']' + cquestion_mark, end='')
            sys.stdout.flush()

        # read user's input
        cresult = sys.stdin.readline().strip()
        if cresult == '' and default is not None:
            cresult = str(default)

        if len(cresult) == 1:
            # check that all the characters are valid
            loop = False
            if valid is not None:
                for c in cresult:
                    if c not in str(valid):
                        print('*** Error: invalid characters found.')
                        print('*** Valid characters are:', valid)
                        print('*** Try again!')
                        loop = True
        else:
            print('*** Error: invalid string length. Try again!')

    return cresult


def readi(prompt, default=None, minval=None, maxval=None,
          allowed_single_chars=None, question_mark=True):
    """Return integer value read from keyboard

    Parameters
    ----------
    prompt : str
        Prompt string.
    default : integer, str or None
        Default value.
    minval :  integer or None
        Mininum allowed value.
    maxval :  integer or None
        Maximum allowed value.
    allowed_single_chars : str
        String containing allowed valid characters.
    question_mark : bool
        If True, display question mark after prompt.

    Returns
    -------
    result : integer
        Read value.

    """

    return read_value(ftype=int,
                      prompt=prompt,
                      default=default,
                      minval=minval,
                      maxval=maxval,
                      allowed_single_chars=allowed_single_chars,
                      question_mark=question_mark)


def readf(prompt, default=None, minval=None, maxval=None,
          allowed_single_chars=None, question_mark=True):
    """Return integer value read from keyboard

    Parameters
    ----------
    prompt : str
        Prompt string.
    default : float or None
        Default value.
    minval :  float or None
        Mininum allowed value.
    maxval :  float or None
        Maximum allowed value.
    allowed_single_chars : str
        String containing allowed valid characters.
    question_mark : bool
        If True, display question mark after prompt.

    Returns
    -------
    result : float
        Read value.

    """

    return read_value(ftype=float,
                      prompt=prompt,
                      default=default,
                      minval=minval,
                      maxval=maxval,
                      allowed_single_chars=allowed_single_chars,
                      question_mark=question_mark)


def read_value(ftype, prompt, default=None, minval=None, maxval=None,
               allowed_single_chars=None, question_mark=True):
    """Return value read from keyboard

    Parameters
    ----------
    ftype : int() or float()
        Function defining the expected type.
    prompt : str
        Prompt string.
    default : int or None
        Default value.
    minval : int or None
        Mininum allowed value.
    maxval : int or None
        Maximum allowed value.
    allowed_single_chars : str
        String containing allowed valid characters.
    question_mark : bool
        If True, display question mark after prompt.

    Returns
    -------
    result : integer, float or str
        Integer, float of single character.

    """

    # avoid PyCharm warning 'might be referenced before assignment'
    result = None

    # question mark
    if question_mark:
        cquestion_mark = ' ? '
    else:
        cquestion_mark = ''

    # check minimum value
    if minval is not None:
        try:
            iminval = ftype(minval)
        except ValueError:
            raise ValueError("'" + str(minval) + "' cannot " +
                             "be used as an minval in readi()")
    else:
        iminval = None

    # check maximum value
    if maxval is not None:
        try:
            imaxval = ftype(maxval)
        except ValueError:
            raise ValueError("'" + str(maxval) + "' cannot " +
                             "be used as an maxval in readi()")
    else:
        imaxval = None

    # minimum and maximum values
    if minval is None and maxval is None:
        cminmax = ''
    elif minval is None:
        cminmax = ' (number <= ' + str(imaxval) + ')'
    elif maxval is None:
        cminmax = ' (number >= ' + str(iminval) + ')'
    else:
        cminmax = ' (' + str(minval) + ' <= number <= ' + str(maxval) + ')'

    # main loop
    loop = True
    while loop:

        # display prompt
        if default is None:
            print(prompt + cminmax + cquestion_mark, end='')
            sys.stdout.flush()
        else:
            print(prompt + cminmax + ' [' + str(default) + ']' +
                  cquestion_mark, end='')
            sys.stdout.flush()

        # read user's input
        cresult = sys.stdin.readline().strip()
        if cresult == '' and default is not None:
            cresult = str(default)

        # if valid allowed single character, return character
        if len(cresult) == 1:
            if allowed_single_chars is not None:
                if cresult in allowed_single_chars:
                    return cresult

        # convert to ftype value
        try:
            result = ftype(cresult)
        except ValueError:
            print("*** Error: invalid " + str(ftype) + " value. Try again!")
        else:
            # check number is within expected range
            if minval is None and maxval is None:
                loop = False
            elif minval is None:
                if result <= imaxval:
                    loop = False
                else:
                    print("*** Error: number out of range. Try again!")
            elif maxval is None:
                if result >= iminval:
                    loop = False
                else:
                    print("*** Error: number out of range. Try again!")
            else:
                if iminval <= result <= imaxval:
                    loop = False
                else:
                    print("*** Error: number out of range. Try again!")

    return result


def main():

    c = readc("1 Enter character", default='y')
    print('>>>', c)
    c = readc("2 Enter character", default='y', valid='ynx')
    print('>>>', c)
    c = readc("3 Enter character", default='oso')
    print('>>>', c)
    c = readc("4 Enter character", default='oso', valid='yn')
    print('>>>', c)

    i = readi("1 Enter integer", default=8)
    print('>>>', i)
    i = readi("2 Enter integer", minval=0)
    print('>>>', i)
    i = readi("3 Enter integer", default=7, maxval=10)
    print('>>>', i)
    i = readi("4 Enter integer", default=101, minval=5, maxval=10)
    print('>>>', i)

    f = readf("1 Enter float", default=8)
    print('>>>', f)
    f = readf("2 Enter float", minval=0)
    print('>>>', f)
    f = readf("3 Enter float", default=7, maxval=10)
    print('>>>', f)
    f = readf("4 Enter float", default=101, minval=5, maxval=10)
    print('>>>', f)


if __name__ == "__main__":

    main()
