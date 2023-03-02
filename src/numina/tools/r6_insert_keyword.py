
import argparse

import astropy.io.fits as fits


def add_key_val(keyname, keyval, keytype, filename, extnum):
    """Add/replace FITS key

    Add/replace the key keyname with value keyval of type keytype in filename.

    Parameters:
    ----------
    keyname : str
        FITS Keyword name.
    keyval : str
        FITS keyword value.
    keytype: str
        FITS keyword type: int, float, str or bool.
    filaname : str
        FITS filename.
    extnum : int
        Extension number where the keyword will be inserted. Note that
        the first extension is number 1 (and not zero).

    """

    funtype = {'int': int, 'float': float, 'str': str, 'bool': bool}
    if keytype not in funtype:
        raise ValueError('Undefined keyword type: ', keytype)
    with fits.open(filename, "update") as hdulist:
        hdulist[extnum].header[keyname] = funtype[keytype](keyval)
        print('>>> Inserting ' + keyname + '=' + keyval + ' in ' + filename)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help='FITS file name (wildcards accepted)',
                        nargs='+')
    parser.add_argument('keyname',
                        help='Keyword name')
    parser.add_argument('keyval',
                        help='keyword value')
    parser.add_argument('--keytype',
                        help='Keyword type (int, float, str, bool)',
                        default='str',
                        choices=['int', 'float', 'str', 'bool'])
    parser.add_argument('--extnum',
                        help='Extension number '
                        '(first extension is 1 and not 0)',
                        default=1, type=int)
    args = parser.parse_args(args=args)

    extnum = args.extnum - 1
    for f in args.filename:
        add_key_val(args.keyname, args.keyval, args.keytype, f, extnum)


if __name__ == '__main__':

    main()
