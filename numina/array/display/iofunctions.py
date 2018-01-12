from __future__ import division
from __future__ import print_function

import sys


def readi(prompt, default=None, minval=None, maxval=None):
    """Return integer value read from keyboard

    Parameters
    ----------
    prompt : str
        Prompt string.
    default : integer or None
        Default value.
    minval :  integer or None
        Mininum allowed value.
    maxval :  integer or None
        Maximum allowed value.

    Returns
    -------
    result : integer
        Read value.

    """

    return read_value(ftype=int,
                      prompt=prompt,
                      default=default,
                      minval=minval,
                      maxval=maxval)


def readf(prompt, default=None, minval=None, maxval=None):
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

    Returns
    -------
    result : float
        Read value.

    """

    return read_value(ftype=float,
                      prompt=prompt,
                      default=default,
                      minval=minval,
                      maxval=maxval)


def read_value(ftype, prompt, default=None, minval=None, maxval=None):
    """Return value read from keyboard

    Parameters
    ----------
    ftype : int() or float()
        Function defining the expected type.
    prompt : str
        Prompt string.
    default : int or None
        Default value
    minval : int or None
        Mininum allowed value
    maxval : int or None
        Maximum allowed value

    Returns
    -------
    result : integer or float
        Integer value

    """

    # avoid PyCharm warning 'might be referenced before assignment'
    result = None

    # check minimum value
    if minval is not None:
        try:
            iminval = ftype(minval)
        except:
            raise ValueError("'" + str(minval) + "' cannot " +
                             "be used as an minval in readi()")
    else:
        iminval = None

    # check maximum value
    if maxval is not None:
        try:
            imaxval = ftype(maxval)
        except:
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
            sys.stdout.write(prompt + cminmax + ' ? ')
        else:
            sys.stdout.write(prompt + cminmax + ' [' + str(default) + '] ? ')

        # read user's input
        cresult = sys.stdin.readline().strip()
        if cresult == '':
            cresult = default

        # convert to integer
        try:
            result = ftype(cresult)
            # check number is within expected range
            if minval is None and maxval is None:
                loop = False
            elif minval is None:
                if result <= imaxval:
                    loop = False
                else:
                    print("Number out of range. Try again!")
            elif maxval is None:
                if result >= iminval:
                    loop = False
                else:
                    print("Number out of range. Try again!")
            else:
                if iminval <= result <= imaxval:
                    loop = False
                else:
                    print("Number out of range. Try again!")
        except:
            print("Invalid value. Try again!")

    return result


def main():

    i = readi("Enter integer", default=8)
    print('>>>', i)
    i = readi("Enter integer", minval=0)
    print('>>>', i)
    i = readi("Enter integer", default=7, maxval=10)
    print('>>>', i)
    i = readi("Enter integer", default=101, minval=5, maxval=10)
    print('>>>', i)

    f = readf("Enter float", default=8)
    print('>>>', f)
    f = readf("Enter float", minval=0)
    print('>>>', f)
    f = readf("Enter float", default=7, maxval=10)
    print('>>>', f)
    f = readf("Enter float", default=101, minval=5, maxval=10)
    print('>>>', f)


if __name__ == "__main__":

    main()
