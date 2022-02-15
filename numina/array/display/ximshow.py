#
# Copyright 2015-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import argparse

import matplotlib
from astropy.io import fits
import numpy as np
import re

from .matplotlib_qt import set_window_geometry
from .pause_debugplot import pause_debugplot
from .fileinfo import list_fileinfo_from_txt
from .fileinfo import check_extnum
from .overplot_ds9reg import overplot_ds9reg
from ..stats import summary

from numina.visualization import ZScaleInterval


GLOBAL_ASPECT = 'auto'
GLOBAL_GEOMETRY = '0,0,800,600'
dum_str = ''  # global variable in function keypress
dum_par = ''  # global variable in function keypress


def check_wavelength_scale(crval1, cdelt1, ctype1, cunit1):
    """Check for wavelength calibration in the X axis.

    Parameters
    ----------
    crval1 : float or None
        CRVAL1 parameter corresponding to wavelength calibration in
        the X direction.
    cdelt1 : float or None
        CDELT1 parameter corresponding to wavelength calibration in
        the X direction.
    ctype1 : str or None
        CTYPE1 parameter corresponding to wavelength calibration in
        the X direction.
    cunit1 : str or None
        CUNIT1 parameter corresponding to wavelength calibration in
        the X direction.

    Returns
    -------
    result : bool
        True in the wavelength calibration has been set.
        False otherwise.

    """
    result = False

    if ctype1 is None and cunit1 is None:
        return result

    if ctype1 is not None:
        if 'wavelength' in ctype1.lower():
            result = True
    if cunit1 is not None:
        if 'angstrom' in cunit1.lower():
            result = True

    if result:
        if crval1 is not None and cdelt1 is not None:
            pass
        else:
            result = False

    return result


def ximshow_jupyter(image2d, **args):
    """Auxiliary function to call ximshow from a jupyter notebook.
    """
    return ximshow(image2d, using_jupyter=True, **args)


def ximshow(image2d, title=None, show=True,
            cbar_label=None, cbar_orientation='None',
            z1z2=None, cmap="hot",
            image_bbox=None, first_pixel=(1, 1),
            aspect=GLOBAL_ASPECT,
            crpix1=None, crval1=None, cdelt1=None, ctype1=None, cunit1=None,
            ds9regfile=None,
            geometry=GLOBAL_GEOMETRY, figuredict=None,
            tight_layout=True,
            debugplot=0, using_jupyter=False):
    """Auxiliary function to display a numpy 2d array.

    Parameters
    ----------
    image2d : 2d numpy array, float
        2d image to be displayed.
    title : string
        Plot title.
    cbar_label : string
        Color bar label.
    cbar_orientation : string
        Color bar orientation: valid options are 'horizontal' or
        'vertical' (or 'None' for no color bar).
    show : bool
        If True, the function shows the displayed image. Otherwise
        the function just invoke the plt.imshow() function and
        plt.show() is expected to be executed outside.
    z1z2 : tuple of floats, string or None
        Background and foreground values. If None, zcuts are employed.
    cmap : string
        Color map to be employed.
    image_bbox : tuple (4 integers)
        Image rectangle to be displayed, with indices given by
        (nc1,nc2,ns1,ns2), which correspond to the numpy array:
        image2d[(ns1-1):ns2,(nc1-1):nc2].
    first_pixel : tuple (2 integers)
        (x0,y0) coordinates of pixel at origin.
    aspect : str
        Control de aspect ratio of the axes. Valid values are 'equal'
        and 'auto'.
    crpix1 : float or None
        CRPIX1 parameter corresponding to wavelength calibration in
        the X direction.
    crval1 : float or None
        CRVAL1 parameter corresponding to wavelength calibration in
        the X direction.
    cdelt1 : float or None
        CDELT1 parameter corresponding to wavelength calibration in
        the X direction.
    ctype1 : str or None
        CTYPE1 parameter corresponding to wavelength calibration in
        the X direction.
    cunit1 : str or None
        CUNIT1 parameter corresponding to wavelength calibration in
        the X direction.
    ds9regfile : file handler
        Ds9 region file to be overplotted.
    geometry : str or None
        x, y, dx, dy values employed to set the window geometry.
    tight_layout : bool
        If True, and show=True, a tight display layout is set.
    figuredict: dictionary
        Parameters for ptl.figure(). Useful for pdf output.
        For example: --figuredict "{'figsize': (8, 10), 'dpi': 100}"
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot
    using_jupyter : bool
        If True, this function is called from a jupyter notebook.

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. This value is returned only when
        'show' is False.

    """

    from numina.array.display.matplotlib_qt import plt

    if not show and using_jupyter:
        plt.ioff()

    # protections
    if not isinstance(image2d, np.ndarray):
        raise ValueError("image2d=" + str(image2d) +
                         " must be a numpy.ndarray")
    elif image2d.ndim != 2:
        raise ValueError("image2d.ndim=" + str(image2d.dim) +
                         " must be 2")

    naxis2_, naxis1_ = image2d.shape

    # check if wavelength calibration is provided
    wavecalib = check_wavelength_scale(
        crval1=crval1, cdelt1=1, ctype1=ctype1, cunit1=cunit1
    )

    # read bounding box limits
    if image_bbox is None:
        nc1 = 1
        nc2 = naxis1_
        ns1 = 1
        ns2 = naxis2_
    else:
        nc1, nc2, ns1, ns2 = image_bbox
        if 1 <= nc1 <= nc2 <= naxis1_:
            pass
        else:
            raise ValueError("Invalid bounding box limits")
        if 1 <= ns1 <= ns2 <= naxis2_:
            pass
        else:
            raise ValueError("Invalid bounding box limits")

    def get_current_zoom(ax_image, debug=False):
        """Return subimage corresponding to current zoom.

        Parameters
        ----------
        ax_image : axes
            Image axes.
        debug : bool
            If True, the image corners are printed.

        Returns
        -------
        subimage : numpy array (floats)
            Subimage.

        """

        xmin_image, xmax_image = ax_image.get_xlim()
        ymin_image, ymax_image = ax_image.get_ylim()
        ixmin = int(xmin_image + 0.5)
        ixmax = int(xmax_image + 0.5)
        iymin = int(ymin_image + 0.5)
        iymax = int(ymax_image + 0.5)
        ixmin -= first_pixel[0] - 1
        ixmax -= first_pixel[0] - 1
        iymin -= first_pixel[1] - 1
        iymax -= first_pixel[1] - 1
        if ixmin < nc1:
            ixmin = nc1
        if ixmin > nc2:
            ixmin = nc2
        if ixmax < nc1:
            ixmax = nc1
        if ixmax > nc2:
            ixmax = nc2
        if iymin < ns1:
            iymin = ns1
        if iymin > ns2:
            iymin = ns2
        if iymax < ns1:
            iymax = ns1
        if iymax > ns2:
            iymax = ns2
        if debug:
            print("\n>>> xmin, xmax, ymin, ymax [pixels; origin (1,1)]:",
                  ixmin, ixmax, iymin, iymax)
        return image2d[(iymin-1):iymax, (ixmin-1):ixmax]

    def keypress(event):
        """Deal with keyboard events, allowing the update of vmin and vmax.

        Note that a call to raw_input() is not allowed within this
        function since, in that case, the following runtime error
        is raised: can't re-enter readline

        For that reason, the new vmin and vmax values should be
        entered blindly.

        To avoid collisions with navigation keyboard shortcuts,
        check the table available at:
        http://matplotlib.org/users/navigation_toolbar.html

        """

        global dum_str
        global dum_par
        if event.key == "?":
            print("""
Keyword events
==============
Home/Reset......................: h or r or home
Back............................: c or left arrow or backspace
Forward.........................: v or right arrow
Pan/Zoom........................: p
Zoom-to-rect....................: o
Save............................: ctrl + s
Toggle fullscreen...............: ctrl + f
Close plot......................: ctrl + w
Set zscale......................: /
Set bg=min and fg=max values....: ,
Display statistical summary.....: ;
Set foreground by keyboard......: m
Set background by keyboard......: n
Activate/deactivate ds9 regions.: a
Change aspect ratio.............: =
Constrain pan/zoom to x axis....: hold x when panning/zooming with mouse
Constrain pan/zoom to y axis....: hold y when panning/zooming with mouse
Preserve aspect ratio...........: hold CONTROL when panning/zooming with mouse
Toggle grid.....................: g when mouse is over an axes
Toggle x axis scale (log/linear): L or k when mouse is over an axes
Toggle y axis scale (log/linear): l when mouse is over an axes
            """)
        elif event.key == "/":
            subimage2d = get_current_zoom(ax, debug=True)
            new_vmin, new_vmax = ZScaleInterval().get_limits(subimage2d)
            print(">>> setting cuts to vmin=" + str(new_vmin) +
                  " and vmax=" + str(new_vmax))
            im_show.set_clim(vmin=new_vmin)
            im_show.set_clim(vmax=new_vmax)
            dum_str = ''
            dum_par = ''
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == ",":
            subimage2d = get_current_zoom(ax, debug=True)
            new_vmin = subimage2d.min()
            new_vmax = subimage2d.max()
            print(">>> setting cuts to vmin=" + str(new_vmin) +
                  " and vmax=" + str(new_vmax))
            im_show.set_clim(vmin=new_vmin)
            im_show.set_clim(vmax=new_vmax)
            dum_str = ''
            dum_par = ''
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == ";":
            subimage2d = get_current_zoom(ax, debug=True)
            summary(subimage2d.flatten(), debug=True)
        elif event.key == "n":
            print("Type (blindly!) vmin <return>")
            dum_str = ''
            dum_par = "vmin"
        elif event.key == "m":
            print("Type (blindly!) vmax <return>")
            dum_str = ''
            dum_par = "vmax"
        elif event.key == "=":
            if ax.get_aspect() == 'equal':
                ax.set_aspect('auto')
            else:
                ax.set_aspect('equal')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == "enter":
            if dum_par == "vmin":
                try:
                    new_vmin = float(dum_str)
                except ValueError:
                    print("Invalid vmin=" + dum_str)
                    dum_str = ''
                    print("Type again (blindly!) vmin <return>")
                else:
                    print("Setting vmin=" + dum_str)
                    im_show.set_clim(vmin=new_vmin)
                    dum_str = ''
                    dum_par = ''
                    plt.show(block=False)
                    plt.pause(0.001)
            elif dum_par == "vmax":
                try:
                    new_vmax = float(dum_str)
                except ValueError:
                    print("Invalid vmax=" + dum_str)
                    dum_str = ''
                    print("Type again (blindly!) vmax <return>")
                else:
                    print("Setting vmax=" + dum_str)
                    im_show.set_clim(vmax=new_vmax)
                    dum_str = ''
                    dum_par = ''
                    plt.show(block=False)
                    plt.pause(0.001)
        else:
            if dum_str == '':
                dum_str = event.key
            else:
                dum_str += event.key

    # plot limits
    xmin = float(nc1) - 0.5 + (first_pixel[0] - 1)
    xmax = float(nc2) + 0.5 + (first_pixel[0] - 1)
    ymin = float(ns1) - 0.5 + (first_pixel[1] - 1)
    ymax = float(ns2) + 0.5 + (first_pixel[1] - 1)

    # display image
    if figuredict is None:
        fig = plt.figure()
    else:
        fig = plt.figure(**figuredict)

    ax = fig.add_subplot(111)
    ax.autoscale(False)
    ax.set_xlabel('image pixel in the X direction')
    ax.set_ylabel('image pixel in the Y direction')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(False)
    if z1z2 is None:
        z1, z2 = ZScaleInterval().get_limits(
            image2d[(ns1 - 1):ns2, (nc1 - 1):nc2]
        )
    elif z1z2 == "minmax":
        z1 = image2d[(ns1 - 1):ns2, (nc1 - 1):nc2].min()
        z2 = image2d[(ns1 - 1):ns2, (nc1 - 1):nc2].max()
    else:
        z1, z2 = z1z2
    im_show = plt.imshow(image2d[(ns1 - 1):ns2, (nc1 - 1):nc2],
                         cmap=cmap, aspect=aspect,
                         vmin=z1, vmax=z2,
                         interpolation='nearest', origin='lower',
                         extent=[xmin, xmax, ymin, ymax])
    if cbar_label is None:
        cbar_label = "Number of counts"
    if cbar_orientation in ["horizontal", "vertical"]:
        plt.colorbar(im_show, shrink=1.0, label=cbar_label,
                     orientation=cbar_orientation)
    if title is not None:
        ax.set_title(title)

    if ds9regfile is not None:
        overplot_ds9reg(ds9regfile.name, ax)

    # set the geometry
    if geometry is not None:
        tmp_str = geometry.split(",")
        x_geom = int(tmp_str[0])
        y_geom = int(tmp_str[1])
        dx_geom = int(tmp_str[2])
        dy_geom = int(tmp_str[3])
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            plt.get_current_fig_manager().resize(x_geom, y_geom)
            plt.get_current_fig_manager().window.wm_geometry(f"+{dx_geom}+{dy_geom}")
        elif backend == 'MacOSX':
            plt.get_current_fig_manager().resize(x_geom, y_geom)
        elif backend == 'Qt5Agg':
            geometry_tuple = x_geom, y_geom, dx_geom, dy_geom
            set_window_geometry(geometry_tuple)
        else:
            pass

    # connect keypress event with function responsible for
    # updating vmin and vmax
    fig.canvas.mpl_connect('key_press_event', keypress)

    # wavelength scale
    if wavecalib:
        if crpix1 is None:
            crpix1 = 1.0
        xminwv = crval1 + (xmin - crpix1) * cdelt1
        xmaxwv = crval1 + (xmax - crpix1) * cdelt1
        ax2 = ax.twiny()
        ax2.grid(False)
        ax2.set_xlim(xminwv, xmaxwv)
        ax2.set_xlabel('Wavelength (Angstroms)')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # show plot or return axes
    if show:
        pause_debugplot(debugplot, pltshow=show, tight_layout=tight_layout)
    else:
        if tight_layout:
            plt.tight_layout()
        # return axes
        if using_jupyter:
            plt.ion()
        return ax


def ximshow_file(singlefile,
                 extnum=1,
                 args_cbar_label=None, args_cbar_orientation='None',
                 args_z1z2=None, args_bbox=None, args_firstpix=None,
                 args_aspect=GLOBAL_ASPECT,
                 args_keystitle=None, args_ds9reg=None,
                 args_geometry=GLOBAL_GEOMETRY, pdf=None,
                 args_figuredict=None,
                 show=True,
                 debugplot=None,
                 using_jupyter=False):
    """Function to execute ximshow() as called from command line.

    Parameters
    ----------
    singlefile : string
        Name of the FITS file to be displayed.
    extnum : int
        Extension number: 1 for first extension (default).
    args_cbar_label : string
        Color bar label.
    args_cbar_orientation : string
        Color bar orientation: valid options are 'horizontal' or
        'vertical' (or 'None' for no color bar).
    args_z1z2 : string or None
        String providing the image cuts tuple: z1, z2, minmax of None
    args_bbox : string or None
        String providing the bounding box tuple: nc1, nc2, ns1, ns2
    args_firstpix : string or None
        String providing the coordinates of lower left pixel.
        args_aspect : str
    args_aspect : str
        Control de aspect ratio of the axes. Valid values are 'equal'
        and 'auto'.
    args_keystitle : string or None
        Tuple of FITS keywords.format: key1,key2,...,keyn.format
    args_ds9reg : file handler
        Ds9 region file to be overplotted.
    args_geometry : string or None
        x, y, dx, dy to define the window geometry. This
        information is ignored if args_pdffile is not None.
    pdf : PdfFile object or None
        If not None, output is sent to PDF file.
    args_figuredict : string containing a dictionary
        Parameters for ptl.figure(). Useful for pdf output.
        For example: --figuredict "{'figsize': (8, 10), 'dpi': 100}"
    show : bool
        If True, the function shows the displayed image. Otherwise
        the function just invoke the plt.imshow() function and
        plt.show() is expected to be executed outside.
    debugplot : integer or None
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.
    using_jupyter : bool
        If True, this function is called from a jupyter notebook.

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. This value is returned only when
        'show' is False.

    """

    # read z1, z2
    if args_z1z2 is None:
        z1z2 = None
    elif args_z1z2 == "minmax":
        z1z2 = "minmax"
    else:
        if args_z1z2[0] == '[':
            tmp_str = args_z1z2[1:]
        else:
            tmp_str = args_z1z2
        tmp_str = re.sub(']', '', tmp_str)
        tmp_str = tmp_str.split(",")
        z1z2 = float(tmp_str[0]), float(tmp_str[1])

    # read input FITS file
    hdulist = fits.open(singlefile)
    if extnum is None or extnum < 1 or extnum > len(hdulist):
        raise ValueError(f'Unexpected extension number {extnum}')
    image_header = hdulist[extnum - 1].header
    image2d = hdulist[extnum - 1].data
    hdulist.close()

    naxis1 = image_header['naxis1']
    if 'naxis2' in image_header:
        naxis2 = image_header['naxis2']
    else:
        naxis2 = 1

    # read wavelength calibration
    if 'crpix1' in image_header:
        crpix1 = image_header['crpix1']
    else:
        crpix1 = None
    if 'crval1' in image_header:
        crval1 = image_header['crval1']
    else:
        crval1 = None
    if 'cdelt1' in image_header:
        cdelt1 = image_header['cdelt1']
    else:
        cdelt1 = None
    if 'ctype1' in image_header:
        ctype1 = image_header['ctype1']
    else:
        ctype1 = None
    if 'cunit1' in image_header:
        cunit1 = image_header['cunit1']
    else:
        cunit1 = None

    # title for plot
    title = singlefile
    if args_keystitle is not None:
        keystitle = args_keystitle
        keysformat = ".".join(keystitle.split(".")[1:])
        keysnames = keystitle.split(".")[0]
        tuple_of_keyval = ()
        for key in keysnames.split(","):
            keyval = image_header[key]
            tuple_of_keyval += (keyval,)
        title += "\n" + str(keysformat % tuple_of_keyval)

    if len(image2d.shape) == 1:
        if image2d.shape != (naxis1,):
            raise ValueError("Unexpected error with NAXIS1")
        image2d = np.reshape(image2d, (1, naxis1))
    elif len(image2d.shape) == 2:
        if image2d.shape != (naxis2, naxis1):
            raise ValueError("Unexpected error with NAXIS1, NAXIS2")
    else:
        raise ValueError("Unexpected number of dimensions > 2")

    print('>>> File..:', singlefile)
    print('>>> NAXIS1:', naxis1)
    print('>>> NAXIS2:', naxis2)

    # read bounding box
    if args_bbox is None:
        nc1 = 1
        nc2 = naxis1
        ns1 = 1
        ns2 = naxis2
    else:
        tmp_bbox = args_bbox.split(",")
        nc1 = int(tmp_bbox[0])
        nc2 = int(tmp_bbox[1])
        ns1 = int(tmp_bbox[2])
        ns2 = int(tmp_bbox[3])
        if nc1 < 1:
            nc1 = 1
        if nc2 > naxis1:
            nc2 = naxis1
        if ns1 < 1:
            ns1 = 1
        if ns2 > naxis2:
            ns2 = naxis2

    # read coordinates of lower left pixel
    if args_firstpix is None:
        nc0 = 1
        ns0 = 1
    else:
        tmp_firstpix = args_firstpix.split(",")
        nc0 = int(tmp_firstpix[0])
        ns0 = int(tmp_firstpix[1])

    # display image
    if args_figuredict is None:
        figuredict = None
    else:
        figuredict = eval(args_figuredict)

    ax = ximshow(image2d=image2d, show=False,
                 cbar_label=args_cbar_label,
                 cbar_orientation=args_cbar_orientation,
                 title=title,
                 z1z2=z1z2,
                 image_bbox=(nc1, nc2, ns1, ns2),
                 first_pixel=(nc0, ns0),
                 aspect=args_aspect,
                 crpix1=crpix1,
                 crval1=crval1,
                 cdelt1=cdelt1,
                 ctype1=ctype1,
                 cunit1=cunit1,
                 ds9regfile=args_ds9reg,
                 geometry=args_geometry,
                 figuredict=figuredict,
                 debugplot=debugplot,
                 using_jupyter=using_jupyter)

    if pdf is not None:
        if show:
            from numina.array.display.matplotlib_qt import plt
            plt.tight_layout()
            pdf.savefig()
        else:
            return ax
    else:
        if show:
            pause_debugplot(debugplot, pltshow=True)
        else:
            # return axes
            return ax


def jimshow(image2d,
            ax=None,
            title=None,
            vmin=None, vmax=None,
            image_bbox=None,
            aspect=GLOBAL_ASPECT,
            xlabel='image pixel in the X direction',
            ylabel='image pixel in the Y direction',
            crpix1=None, crval1=None, cdelt1=None, ctype1=None, cunit1=None,
            grid=False,
            cmap='hot',
            cbar_label='Number of counts',
            cbar_orientation='horizontal'):
    """Auxiliary function to display a numpy 2d array via axes object.

    Parameters
    ----------
    image2d : 2d numpy array, float
        2d image to be displayed.
    ax : axes object
        Matplotlib axes instance. Note that this value is also
        employed as output.
    title : string
        Plot title.
    vmin : float, 'min', or None
        Background value. If None, the minimum zcut is employed.
    vmax : float, 'max', or None
        Foreground value. If None, the maximum zcut is employed.
    image_bbox : tuple (4 integers)
        Image rectangle to be displayed, with indices given by
        (nc1,nc2,ns1,ns2), which correspond to the numpy array:
        image2d[(ns1-1):ns2,(nc1-1):nc2].
    aspect : str
        Control de aspect ratio of the axes. Valid values are 'equal'
        and 'auto'.
    xlabel : string
        X-axis label.
    ylabel : string
        Y-axis label.
    crpix1 : float or None
        CRPIX1 parameter corresponding to wavelength calibration in
        the X direction.
    crval1 : float or None
        CRVAL1 parameter corresponding to wavelength calibration in
        the X direction.
    cdelt1 : float or None
        CDELT1 parameter corresponding to wavelength calibration in
        the X direction.
    ctype1 : str or None
        CTYPE1 parameter corresponding to wavelength calibration in
        the X direction.
    cunit1 : str or None
        CUNIT1 parameter corresponding to wavelength calibration in
        the X direction.
    grid : bool
        If True, overplot grid.
    cmap : string
        Color map to be employed.
    cbar_label : string
        Color bar label.
    cbar_orientation : string
        Color bar orientation: valid options are 'horizontal' or
        'vertical' (or 'None' for no color bar).

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. Note that this value must also
        be provided as input.

    """

    if ax is None:
        raise ValueError('ax=None is not valid in this function')

    naxis2_, naxis1_ = image2d.shape

    if image_bbox is None:
        nc1, nc2, ns1, ns2 = 1, naxis1_, 1, naxis2_
    else:
        nc1, nc2, ns1, ns2 = image_bbox
    if 1 <= nc1 <= nc2 <= naxis1_:
        pass
    else:
        raise ValueError("Invalid bounding box limits")
    if 1 <= ns1 <= ns2 <= naxis2_:
        pass
    else:
        raise ValueError("Invalid bounding box limits")

    # plot limits
    xmin = float(nc1) - 0.5
    xmax = float(nc2) + 0.5
    ymin = float(ns1) - 0.5
    ymax = float(ns2) + 0.5

    image2d_region = image2d[(ns1 - 1):ns2, (nc1 - 1):nc2]

    if vmin is None or vmax is None:
        z1, z2 = ZScaleInterval().get_limits(image2d_region)
    else:
        z1, z2 = None, None

    if vmin is None:
        vmin = z1
    elif vmin == 'min':
        vmin = image2d_region.min()
    if vmax is None:
        vmax = z2
    elif vmax == 'max':
        vmax = image2d_region.max()

    im_show = ax.imshow(
        image2d_region,
        cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax,
        interpolation="nearest", origin="lower",
        extent=[xmin, xmax, ymin, ymax]
    )
    if cbar_orientation in ['horizontal', 'vertical']:
        import matplotlib.pyplot as plt
        plt.colorbar(im_show, shrink=1.0,
                     label=cbar_label, orientation=cbar_orientation,
                     ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)
    if title is not None:
        ax.set_title(title)

    wavecalib = check_wavelength_scale(
        crval1=crval1, cdelt1=1, ctype1=ctype1, cunit1=cunit1
    )
    if wavecalib:
        if crpix1 is None:
            crpix1 = 1.0
        xminwv = crval1 + (xmin - crpix1) * cdelt1
        xmaxwv = crval1 + (xmax - crpix1) * cdelt1
        ax2 = ax.twiny()
        ax2.grid(False)
        ax2.set_xlim(xminwv, xmaxwv)
        ax2.set_xlabel('Wavelength (Angstroms)')


def jimshowfile(filename,
                extnum=1,
                ax=None,
                title=None,
                vmin=None, vmax=None,
                image_bbox=None,
                aspect=GLOBAL_ASPECT,
                xlabel='image pixel in the X direction',
                ylabel='image pixel in the Y direction',
                crpix1=None, crval1=None, cdelt1=None, ctype1=None, cunit1=None,
                grid=False,
                cmap='hot',
                cbar_label='Number of counts',
                cbar_orientation='horizontal'):
    """Auxiliary function to display a FITS image via axes object.

    Parameters
    ----------
    filename : string
        Input FITS file name.
    extnum : int
        Extension number (1: primary)
    ax : axes object
        Matplotlib axes instance. Note that this value is also
        employed as output.
    title : string
        Plot title.
    vmin : float, 'min', or None
        Background value. If None, the minimum zcut is employed.
    vmax : float, 'max', or None
        Foreground value. If None, the maximum zcut is employed.
    image_bbox : tuple (4 integers)
        Image rectangle to be displayed, with indices given by
        (nc1,nc2,ns1,ns2), which correspond to the numpy array:
        image2d[(ns1-1):ns2,(nc1-1):nc2].
    aspect : str
        Control de aspect ratio of the axes. Valid values are 'equal'
        and 'auto'.
    xlabel : string
        X-axis label.
    ylabel : string
        Y-axis label.
    crpix1 : float or None
        CRPIX1 parameter corresponding to wavelength calibration in
        the X direction.
    crval1 : float or None
        CRVAL1 parameter corresponding to wavelength calibration in
        the X direction.
    cdelt1 : float or None
        CDELT1 parameter corresponding to wavelength calibration in
        the X direction.
    ctype1 : str or None
        CTYPE1 parameter corresponding to wavelength calibration in
        the X direction.
    cunit1 : str or None
        CUNIT1 parameter corresponding to wavelength calibration in
        the X direction.
    grid : bool
        If True, overplot grid.
    cmap : string
        Color map to be employed.
    cbar_label : string
        Color bar label.
    cbar_orientation : string
        Color bar orientation: valid options are 'horizontal' or
        'vertical' (or 'None' for no color bar).

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. Note that this value must also
        be provided as input.

    """

    # read input FITS file
    hdulist = fits.open(filename)
    if extnum is None or extnum < 1 or extnum > len(hdulist):
        raise ValueError(f'Unexpected extension number {extnum}')
    image2d = hdulist[extnum - 1].data
    hdulist.close()

    return jimshow(image2d,
                   ax=ax,
                   title=title,
                   vmin=vmin, vmax=vmax,
                   image_bbox=image_bbox,
                   aspect=aspect,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   crpix1=crpix1, crval1=crval1, cdelt1=cdelt1,
                   ctype1=ctype1, cunit1=cunit1,
                   grid=grid,
                   cmap=cmap,
                   cbar_label=cbar_label,
                   cbar_orientation=cbar_orientation)


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(
        description='description: display FITS images'
    )

    # positional arguments
    parser.add_argument("filename",
                        help="FITS file (wildcards allowed) "
                             "or txt file with list of FITS files",
                        nargs="+")

    # optional arguments
    parser.add_argument('--extnum',
                        help='Extension number in input files (note that ' +
                             'first extension is 1 = default value)',
                        default=1, type=int)
    parser.add_argument("--z1z2",
                        help="tuple [z1,z2], minmax or None (use zscale)",
                        type=str)
    parser.add_argument("--bbox",
                        help="bounding box tuple: nc1,nc2,ns1,ns2")
    parser.add_argument("--firstpix",
                        help="coordinates of lower left pixel: nc0, ns0")
    parser.add_argument("--aspect",
                        help="aspect ratio (equal or auto)",
                        type=str,
                        choices=['equal', 'auto'], default=GLOBAL_ASPECT)
    parser.add_argument("--cbar_label",
                        help="color bar label",
                        type=str, default='Number of counts')
    parser.add_argument("--cbar_orientation",
                        help="color bar orientation",
                        type=str,
                        choices=['horizontal', 'vertical', 'None'],
                        default='horizontal')
    parser.add_argument("--keystitle",
                        help="tuple of FITS keywords.format: " +
                             "key1,key2,...keyn.'format'")
    parser.add_argument("--ds9reg",
                        help="ds9 region file to be overplotted",
                        type=argparse.FileType('rt'))
    parser.add_argument("--geometry",
                        help='string "x,y,dx,dy"',
                        default=GLOBAL_GEOMETRY)
    parser.add_argument("--pdffile",
                        help="ouput PDF file name",
                        type=argparse.FileType('w'))
    parser.add_argument("--figuredict",
                        help="string with dictionary of parameters for"
                             "plt.figure()",
                        type=str)
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=12)",
                        default=12, type=int,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])
    args = parser.parse_args(args)

    if abs(args.debugplot) in [21, 22]:
        print('>> args.filename: ', args.filename)

    if len(args.filename) == 1:
        list_fits_files = []
        list_extnum = []
        for tmp in list_fileinfo_from_txt(args.filename[0]):
            list_fits_files.append(tmp.filename)
            list_extnum.append(tmp.extnum)
    else:
        list_fits_files = []
        list_extnum = []
        for tmp in args.filename:
            tmpfile, tmpextnum = check_extnum(tmp)
            for tmptmp in list_fileinfo_from_txt(tmpfile):
                list_fits_files.append(tmptmp.filename)
                list_extnum.append(tmpextnum)

    list_extnum = [args.extnum if dum is None else dum for dum in list_extnum]

    if abs(args.debugplot) in [21, 22]:
        print('>> Filenames.: ', list_fits_files)
        print('>> Extensions: ', list_extnum)

    # read pdffile
    if args.pdffile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(args.pdffile.name)
    else:
        from numina.array.display.matplotlib_qt import plt
        pdf = None

    for myfile, extnum in zip(list_fits_files, list_extnum):
        if extnum is None:
            extnum = args.extnum
        ximshow_file(singlefile=myfile,
                     extnum=extnum,
                     args_z1z2=args.z1z2,
                     args_bbox=args.bbox,
                     args_firstpix=args.firstpix,
                     args_aspect=args.aspect,
                     args_cbar_label=args.cbar_label,
                     args_cbar_orientation=args.cbar_orientation,
                     args_keystitle=args.keystitle,
                     args_ds9reg=args.ds9reg,
                     args_geometry=args.geometry,
                     pdf=pdf,
                     args_figuredict=args.figuredict,
                     debugplot=args.debugplot)

    if pdf is not None:
        pdf.close()


if __name__ == "__main__":

    main()
