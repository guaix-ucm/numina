
import argparse

import astropy.io.fits as fits


def insert_image(filename, extnum_filename, auximage, extnum_auximage):
    """Replace image in filename by another image (same size) in newimage.

    Parameters
    ----------
    filename : str
        File name where the new image will be inserted.
    extnum_filename : int
        Extension number in filename where the new image will be
        inserted. Note that the first extension is 1 (and not zero).
    auximage : str
        File name of the new image.
    extnum_auximage : int
        Extension number where the new image is located in auximage.
        Note that the first extension is 1 (and not zero).

    """

    # read the new image
    with fits.open(auximage) as hdulist:
        newimage = hdulist[extnum_auximage].data

    # open the destination image
    hdulist = fits.open(filename, mode='update')
    oldimage_shape = hdulist[extnum_filename].data.shape
    if oldimage_shape == newimage.shape:
        hdulist[extnum_filename].data = newimage
        hdulist.flush()
    else:
        print('filename shape:', oldimage_shape)
        print('newimage shape:', newimage.shape)
        print("ERROR: new image doesn't have the same shape")

    hdulist.close()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="FITS file name")
    parser.add_argument("auximage",
                        help="image")
    parser.add_argument("--extnum_filename",
                        help="extension number (first=1)",
                        default=1, type=int)
    parser.add_argument("--extnum_auximage",
                        help="extension number in auximage (first=1)",
                        default=1, type=int)
    args = parser.parse_args(args=args)

    insert_image(args.filename, args.extnum_filename - 1,
                 args.auximage, args.extnum_auximage - 1)


if __name__ == '__main__':

    main()
