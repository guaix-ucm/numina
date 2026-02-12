#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
from astropy.units import Quantity
import astropy.units as u
from joblib import Parallel, delayed
import json
import logging
import numpy as np
import os
from pathlib import Path
from rich.pretty import pretty_repr
import time
import yaml

from .define_3d_wcs import get_wvparam_from_wcs3d
from .define_3d_wcs import wcs_to_header_using_cd_keywords
from .display_skycalc import display_skycalc
from .compute_image2d_rss_from_detector_method1 import compute_image2d_rss_from_detector_method1
from .compute_image3d_ifu_from_rss_method1 import compute_image3d_ifu_from_rss_method1
from .generate_image2d_method0_ifu import generate_image2d_method0_ifu
from .generate_geometry_for_scene_block import generate_geometry_for_scene_block
from .generate_image3d_method0_ifu import generate_image3d_method0_ifu
from .generate_spectrum_for_scene_block import generate_spectrum_for_scene_block
from .load_atmosphere_transmission_curve import load_atmosphere_transmission_curve
from .raise_valueerror import raise_ValueError
from .save_image2d_detector_method0 import save_image2d_detector_method0
from .save_image2d_rss import save_image2d_rss
from .set_wavelength_unit_and_range import set_wavelength_unit_and_range
from .update_image2d_rss_detector_method0 import update_image2d_rss_detector_method0


def ifu_simulator(
    wcs3d,
    header_keys,
    naxis1_detector,
    naxis2_detector,
    nslices,
    noversampling_whitelight,
    scene_fname,
    flux_factor,
    seeing_fwhm_arcsec,
    seeing_psf,
    instrument_pa,
    airmass,
    parallactic_angle,
    flatpix2pix,
    atmosphere_transmission,
    bias,
    rnoise,
    spectral_blurring_pixel,
    faux_dict,
    rng,
    parallel_computation,
    prefix_intermediate_fits,
    stop_after_ifu_3D_method0=False,
    logger=None,
    instname=None,
    subtitle=None,
    plots=False,
    output_dir=".",
):
    """IFU simulator.

    This function does not return any value. The main output of this function
    is a set of FITS files with intermediate results, which are stored in
    the output directory defined by the user.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    header_keys : `~astropy.io.fits.header.Header`
        FITS header with additional keywords to be merged together with
        the WCS information.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1, dispersion direction.
    naxis2_detector : `~astropy.units.Quantity`
        Detector NAXIS2, spatial direction (slices).
    nslices : int
        Number of IFU slices.
    noversampling_whitelight : int
        Oversampling factor (integer number) to generate the white
        light image.
    scene_fname : str
        YAML scene file name.
    flux_factor : float
        Flux factor to be applied to the number of photons defined
        in the scene file.
    seeing_fwhm_arcsec : `~astropy.units.Quantity`
        Seeing FWHM.
    seeing_psf : str
        Seeing PSF.
    instrument_pa : `~astropy.units.Quantity`
        Instrument position angle.
    airmass : float
        Airmass.
    parallactic_angle : `~astropy.units.Quantity`
        Parallactic angle. This number must be within the range
        [-90,+90] deg.
    flatpix2pix : str
        String indicating whether a pixel-to-pixel flatfield is
        applied or not. Two possible values:
        - 'default': use default flatfield defined in 'faux_dict'
        - 'none': do not apply flatfield
    atmosphere_transmission : str
        String indicating whether the atmosphere transmission of
        the atmosphere is applied or not. Two possible values are:
        - 'default': use default curve defined in 'faux_dict'
        - 'none': do not apply atmosphere transmission
    bias : `~astropy.units.Quantity`
        Bias level (in ADU).
    rnoise : `~astropy.units.Quantity`
        Readout noise standard deviation (in ADU). Assumed to be
        Gaussian.
    spectral_blurring_pixel : `~astropy.units.Quantity`
        Spectral blurring (in pixels) to be introduced when generating
        the initial RSS image from the original 3D data cube.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    parallel_computation : bool
        It True, use parallel processing.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    stop_after_ifu_3D_method0 : bool
        If True, stop execution after generating ifu_3D_method0 image.
    logger : logging.Logger or None, optional
        Logger for logging messages. If None, a default logger will be used.
    instname : str or None
        Instrument name.
    subtitle : str or None
        Plot subtitle.
    plots : bool
        If True, plot intermediate results.
    output_dir : str or `~pathlib.Path`, optional
        Output directory to store results.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(" ")
        for item in faux_dict:
            logger.debug(f"- Required file for item {item}: {faux_dict[item]}")

    # check output_dir is a valid directory
    if not os.path.isdir(output_dir):
        raise_ValueError(f"Invalid output directory: {output_dir}")
    else:
        logger.debug(f"Output directory: {output_dir}")

    if plots:
        # display SKYCALC predictions for sky radiance and transmission
        display_skycalc(faux_skycalc=faux_dict["skycalc"])

    # spatial IFU limits
    naxis1_ifu = Quantity(value=wcs3d.array_shape[2], unit=u.pix, dtype=int)
    naxis2_ifu = Quantity(value=wcs3d.array_shape[1], unit=u.pix, dtype=int)
    min_x_ifu = 0.5 * u.pix
    max_x_ifu = naxis1_ifu + 0.5 * u.pix
    min_y_ifu = 0.5 * u.pix
    max_y_ifu = naxis2_ifu + 0.5 * u.pix

    # wavelength limits
    wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)
    wmin = wv_crval1 + (0.5 * u.pix - wv_crpix1) * wv_cdelt1
    wmax = wmin + naxis1_detector * wv_cdelt1
    reference_wave_vacuum_differential_refraction = (wmin + wmax) / 2

    # load atmosphere transmission curve
    wave_transmission, curve_transmission = load_atmosphere_transmission_curve(
        atmosphere_transmission=atmosphere_transmission,
        wmin=wmin,
        wmax=wmax,
        wv_cunit1=wv_cunit1,
        faux_dict=faux_dict,
        logger=logger,
    )

    required_keys_in_scene_block = {
        "scene_block_name",
        "spectrum",
        "geometry",
        "nphotons",
        "wavelength_sampling",  # default: random
        "apply_atmosphere_transmission",  # default: True
        "apply_seeing",  # default: True
        "render",  # default: True
    }

    nphotons_all = 0
    simulated_wave_all = None
    simulated_x_ifu_all = None
    simulated_y_ifu_all = None

    # main loop (rendering of scene blocks)
    with open(scene_fname, "rt") as fstream:
        scene_dict = yaml.safe_load_all(fstream)
        for scene_block in scene_dict:
            if "scene_block_name" not in scene_block.keys():
                logger.info(" ")
                logger.info(f"[red]ERROR while processing {scene_fname}[/red]")
                raise_ValueError("key scene_block_name not found!")
            scene_block_name = scene_block["scene_block_name"]
            logger.info(f"[green]* Processing: {scene_block_name}[/green]")
            # insert default values for keys not provided
            if "wavelength_sampling" not in scene_block.keys():
                scene_block["wavelength_sampling"] = "random"
                logger.info(f"[cyan]WARNING: asumming {scene_block['wavelength_sampling']=}[/cyan]")
            if "render" not in scene_block.keys():
                scene_block["render"] = True
                logger.info(f"[cyan]WARNING: asumming {scene_block['render']=}[/cyan]")
            if "apply_atmosphere_transmission" not in scene_block.keys():
                scene_block["apply_atmosphere_transmission"] = True
                logger.info(f"[cyan]WARNING: asumming {scene_block['apply_atmosphere_transmission']=}[/cyan]")
            if "apply_seeing" not in scene_block.keys():
                scene_block["apply_seeing"] = True
                logger.info(f"[cyan]WARNING: asumming {scene_block['apply_seeing']=}[/cyan]")
            scene_block_keys = set(scene_block.keys())
            # insert default values
            if scene_block_keys != required_keys_in_scene_block:
                logger.info(f"[red]ERROR while processing: {scene_fname}[/red]")
                logger.info(f"[blue]expected keys..: [/blue]{required_keys_in_scene_block}")
                logger.info(f"[blue]keys found.....: [/blue]{scene_block_keys}")
                list_unexpected_keys = list(scene_block_keys.difference(required_keys_in_scene_block))
                if len(list_unexpected_keys) > 0:
                    logger.info(f"[red]unexpected keys: [/red]{list_unexpected_keys}")
                list_missing_keys = list(required_keys_in_scene_block.difference(scene_block_keys))
                if len(list_missing_keys) > 0:
                    logger.info(f"[red]missing keys...: [/red]{list_missing_keys}")
                logger.info(f"scene_block={pretty_repr(scene_block)}")
                raise_ValueError(f"Invalid format in file: {scene_fname}")
            logger.debug(f"scene_block={pretty_repr(scene_block)}")

            nphotons = int(float(scene_block["nphotons"]) * flux_factor)
            wavelength_sampling = scene_block["wavelength_sampling"]
            if wavelength_sampling not in ["random", "fixed"]:
                raise_ValueError(f"Unexpected {wavelength_sampling=}")
            apply_atmosphere_transmission = scene_block["apply_atmosphere_transmission"]
            if atmosphere_transmission == "none" and apply_atmosphere_transmission:
                logger.info(f"[cyan]WARNING: {apply_atmosphere_transmission=} when {atmosphere_transmission=}[/cyan]")
                logger.info(f"{atmosphere_transmission=} overrides {apply_atmosphere_transmission=}")
                logger.info("The atmosphere transmission will not be applied!")
                apply_atmosphere_transmission = False
            apply_seeing = scene_block["apply_seeing"]
            if apply_seeing:
                if seeing_fwhm_arcsec.value < 0:
                    raise_ValueError(f"Unexpected {seeing_fwhm_arcsec=}")
                elif seeing_fwhm_arcsec == 0:
                    logger.info(f"[cyan]WARNING: {apply_seeing=} when {seeing_fwhm_arcsec=}[/cyan]")
                    logger.info("Seeing effect will not be applied!")
                    apply_seeing = False
            render = scene_block["render"]
            if nphotons > 0 and render:
                # set wavelength unit and range
                wave_unit, wave_min, wave_max = set_wavelength_unit_and_range(
                    scene_fname=scene_fname,
                    scene_block=scene_block,
                    wmin=wmin,
                    wmax=wmax,
                    logger=logger,
                )
                # generate spectrum
                simulated_wave = generate_spectrum_for_scene_block(
                    scene_fname=scene_fname,
                    scene_block=scene_block,
                    faux_dict=faux_dict,
                    wave_unit=wave_unit,
                    wave_min=wave_min,
                    wave_max=wave_max,
                    nphotons=nphotons,
                    wavelength_sampling=wavelength_sampling,
                    apply_atmosphere_transmission=apply_atmosphere_transmission,
                    wave_transmission=wave_transmission,
                    curve_transmission=curve_transmission,
                    rng=rng,
                    naxis1_detector=naxis1_detector,
                    logger=logger,
                    plots=plots,
                )
                # convert to default wavelength_unit
                simulated_wave = simulated_wave.to(wv_cunit1)
                # distribute photons in the IFU focal plane
                simulated_x_ifu, simulated_y_ifu = generate_geometry_for_scene_block(
                    scene_fname=scene_fname,
                    scene_block=scene_block,
                    nphotons=nphotons,
                    apply_seeing=apply_seeing,
                    seeing_fwhm_arcsec=seeing_fwhm_arcsec,
                    seeing_psf=seeing_psf,
                    airmass=airmass,
                    parallactic_angle=parallactic_angle,
                    reference_wave_vacuum_differential_refraction=reference_wave_vacuum_differential_refraction,
                    simulated_wave=simulated_wave,
                    instrument_pa=instrument_pa,
                    wcs3d=wcs3d,
                    min_x_ifu=min_x_ifu,
                    max_x_ifu=max_x_ifu,
                    min_y_ifu=min_y_ifu,
                    max_y_ifu=max_y_ifu,
                    nslices=nslices,
                    rng=rng,
                    logger=logger,
                    plots=plots,
                )
                # store all simulated photons
                if nphotons_all == 0:
                    simulated_wave_all = simulated_wave
                    simulated_x_ifu_all = simulated_x_ifu
                    simulated_y_ifu_all = simulated_y_ifu
                else:
                    simulated_wave_all = np.concatenate((simulated_wave_all, simulated_wave))
                    simulated_x_ifu_all = np.concatenate((simulated_x_ifu_all, simulated_x_ifu))
                    simulated_y_ifu_all = np.concatenate((simulated_y_ifu_all, simulated_y_ifu))
                # ---
                # update nphotons
                logger.debug(f"[blue]--> {nphotons} photons simulated[/blue]")
                if nphotons_all == 0:
                    nphotons_all = nphotons
                else:
                    nphotons_all += nphotons
                if (
                    len({nphotons_all, len(simulated_wave_all), len(simulated_x_ifu_all), len(simulated_y_ifu_all)})
                    != 1
                ):
                    logger.info(f"[red]ERROR: check the following numbers:[/red]")
                    logger.info(f"{nphotons_all=}")
                    logger.info(f"{len(simulated_wave_all)=}")
                    logger.info(f"{len(simulated_x_ifu_all)=}")
                    logger.info(f"{len(simulated_y_ifu_all)=}")
                    raise_ValueError("Unexpected differences found in the previous numbers")
            else:
                if nphotons == 0:
                    logger.warning(f"[cyan]WARNING -> nphotons: 0[/cyan]")
                else:
                    logger.warning(f"[cyan]WARNING -> render: False[/cyan]")

    # filter simulated photons to keep only those that fall within
    # the IFU field of view and within the expected spectral range
    # (note that this step also removes simulated photons with
    # negative wavelength value corresponding to those absorbed by
    # the atmosphere when applying the transmission curve)
    textwidth_nphotons_number = len(str(nphotons_all))
    if nphotons_all > 0:
        logger.debug("Filtering photons within IFU field of view and spectral range...")
        logger.debug(f"Initial number of simulated photons: {nphotons_all:>{textwidth_nphotons_number}}")
        cond1 = simulated_x_ifu_all >= min_x_ifu
        cond2 = simulated_x_ifu_all <= max_x_ifu
        cond3 = simulated_y_ifu_all >= min_y_ifu
        cond4 = simulated_y_ifu_all <= max_y_ifu
        cond5 = simulated_wave_all >= wmin
        cond6 = simulated_wave_all <= wmax
        iok = np.where(cond1 & cond2 & cond3 & cond4 & cond5 & cond6)[0]
    else:
        iok = np.array([], dtype=int)

    if len(iok) < nphotons_all:
        simulated_x_ifu_all = simulated_x_ifu_all[iok]
        simulated_y_ifu_all = simulated_y_ifu_all[iok]
        simulated_wave_all = simulated_wave_all[iok]
        nphotons_all = len(iok)
    logger.debug(f"[blue]Final number of simulated photons..: {nphotons_all:>{textwidth_nphotons_number}}[/blue]")

    # ---------------------------------------------------------------
    # compute image2d IFU, white image, with and without oversampling
    # ---------------------------------------------------------------
    logger.debug(f"[green]* Computing image2d IFU (method 0) with and without oversampling[/green]")
    for noversampling in {noversampling_whitelight, 1}:
        generate_image2d_method0_ifu(
            wcs3d=wcs3d,
            header_keys=header_keys,
            noversampling_whitelight=noversampling,
            simulated_x_ifu_all=simulated_x_ifu_all,
            simulated_y_ifu_all=simulated_y_ifu_all,
            prefix_intermediate_fits=prefix_intermediate_fits,
            instname=instname,
            subtitle=subtitle,
            scene=scene_fname,
            plots=plots,
            logger=logger,
            output_dir=output_dir,
        )

    # ----------------------------
    # compute image3d IFU, method0
    # ----------------------------
    logger.debug(f"[green]* Computing image3d IFU (method 0)[/green]")
    bins_x_ifu = (0.5 + np.arange(naxis1_ifu.value + 1)) * u.pix
    bins_y_ifu = (0.5 + np.arange(naxis2_ifu.value + 1)) * u.pix
    bins_wave = (
        wv_crval1
        + ((np.arange(naxis2_detector.value + 1) + 1) * u.pix - wv_crpix1) * wv_cdelt1
        - 0.5 * u.pix * wv_cdelt1
    )
    generate_image3d_method0_ifu(
        wcs3d=wcs3d,
        header_keys=header_keys,
        simulated_x_ifu_all=simulated_x_ifu_all,
        simulated_y_ifu_all=simulated_y_ifu_all,
        simulated_wave_all=simulated_wave_all,
        bins_x_ifu=bins_x_ifu,
        bins_y_ifu=bins_y_ifu,
        bins_wave=bins_wave,
        prefix_intermediate_fits=prefix_intermediate_fits,
        logger=logger,
        output_dir=output_dir,
    )

    if stop_after_ifu_3D_method0:
        raise SystemExit(f"Program stopped because {stop_after_ifu_3D_method0=}")

    # --------------------------------------------
    # compute image2d RSS and in detector, method0
    # --------------------------------------------
    logger.debug(f"[green]* Computing image2d RSS and detector (method 0)[/green]")
    bins_x_detector = np.linspace(start=0.5, stop=naxis1_detector.value + 0.5, num=naxis1_detector.value + 1)
    bins_y_detector = np.linspace(start=0.5, stop=naxis2_detector.value + 0.5, num=naxis2_detector.value + 1)

    # read ifu2detector transformations
    dict_ifu2detector = json.loads(open(faux_dict["model_ifu2detector"], mode="rt").read())

    # additional degradation in the spectral direction
    # (in units of detector pixels)
    if spectral_blurring_pixel.unit != u.pix:
        raise_ValueError(f"Unexpected unit for {spectral_blurring_pixel.unit=}")
    extra_degradation_spectral_direction = (
        rng.normal(loc=0.0, scale=spectral_blurring_pixel.value, size=nphotons_all) * u.pix
    )

    # initialize images
    image2d_rss_method0 = np.zeros((naxis1_ifu.value * nslices, naxis1_detector.value))
    image2d_detector_method0 = np.zeros((naxis2_detector.value, naxis1_detector.value))

    # update images
    # (accelerate computation using joblib.Parallel)
    t0 = time.time()
    if not parallel_computation:
        # explicit loop in slices
        for islice in range(nslices):
            logger.debug(f"{islice=}")
            update_image2d_rss_detector_method0(
                islice=islice,
                simulated_x_ifu_all=simulated_x_ifu_all,
                simulated_y_ifu_all=simulated_y_ifu_all,
                simulated_wave_all=simulated_wave_all,
                naxis1_ifu=naxis1_ifu,
                bins_x_ifu=bins_x_ifu,
                bins_wave=bins_wave,
                bins_x_detector=bins_x_detector,
                bins_y_detector=bins_y_detector,
                wv_cdelt1=wv_cdelt1,
                extra_degradation_spectral_direction=extra_degradation_spectral_direction,
                dict_ifu2detector=dict_ifu2detector,
                image2d_rss_method0=image2d_rss_method0,
                image2d_detector_method0=image2d_detector_method0,
            )
    else:
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(update_image2d_rss_detector_method0)(
                islice=islice,
                simulated_x_ifu_all=simulated_x_ifu_all,
                simulated_y_ifu_all=simulated_y_ifu_all,
                simulated_wave_all=simulated_wave_all,
                naxis1_ifu=naxis1_ifu,
                bins_x_ifu=bins_x_ifu,
                bins_wave=bins_wave,
                bins_x_detector=bins_x_detector,
                bins_y_detector=bins_y_detector,
                wv_cdelt1=wv_cdelt1,
                extra_degradation_spectral_direction=extra_degradation_spectral_direction,
                dict_ifu2detector=dict_ifu2detector,
                image2d_rss_method0=image2d_rss_method0,
                image2d_detector_method0=image2d_detector_method0,
            )
            for islice in range(nslices)
        )
    t1 = time.time()
    logger.debug(f"Delta time: {t1 - t0}")

    # save RSS image (note that the flatfield effect is not included!)
    save_image2d_rss(
        wcs3d=wcs3d,
        header_keys=header_keys,
        image2d_rss=image2d_rss_method0,
        method=0,
        prefix_intermediate_fits=prefix_intermediate_fits,
        bitpix=16,
        logger=logger,
        output_dir=output_dir,
    )

    # apply flatpix2pix to detector image
    if flatpix2pix not in ["default", "none"]:
        raise_ValueError(f"Invalid {flatpix2pix=}")
    if flatpix2pix == "default":
        infile = faux_dict["flatpix2pix"]
        with fits.open(infile) as hdul:
            image2d_flatpix2pix = hdul[0].data
        if np.min(image2d_flatpix2pix) <= 0:
            logger.info(f"- minimum flatpix2pix value: {np.min(image2d_flatpix2pix)}")
            raise_ValueError("Unexpected signal in flatpix2pix <= 0")
        naxis2_flatpix2pix, naxis1_flatpix2pix = image2d_flatpix2pix.shape
        naxis1_flatpix2pix *= u.pix
        naxis2_flatpix2pix *= u.pix
        if (naxis1_flatpix2pix != naxis1_detector) or (naxis2_flatpix2pix != naxis2_detector):
            raise_ValueError(f"Unexpected flatpix2pix shape: naxis1={naxis1_flatpix2pix}, naxis2={naxis2_flatpix2pix}")
        logger.debug(f"Applying flatpix2pix: {os.path.basename(infile)} to detector image")
        logger.debug(f"- minimum flatpix2pix value: {np.min(image2d_flatpix2pix):.6f}")
        logger.debug(f"- maximum flatpix2pix value: {np.max(image2d_flatpix2pix):.6f}")
        image2d_detector_method0 /= image2d_flatpix2pix
    else:
        logger.debug("Skipping applying flatpix2pix")

    # apply bias to detector image
    if bias.value > 0:
        image2d_detector_method0 += bias.value
        logger.debug(f"Applying {bias=} to detector image")
    else:
        logger.debug("Skipping adding bias")

    # apply Gaussian readout noise to detector image
    if rnoise.value > 0:
        logger.debug(f"Applying Gaussian {rnoise=} to detector image")
        ntot_pixels = naxis1_detector.value * naxis2_detector.value
        image2d_rnoise_flatten = rng.normal(loc=0.0, scale=rnoise.value, size=ntot_pixels)
        image2d_detector_method0 += image2d_rnoise_flatten.reshape((naxis2_detector.value, naxis1_detector.value))
    else:
        logger.debug("Skipping adding Gaussian readout noise")

    save_image2d_detector_method0(
        header_keys,
        image2d_detector_method0=image2d_detector_method0,
        prefix_intermediate_fits=prefix_intermediate_fits,
        bitpix=16,
        logger=logger,
        output_dir=output_dir,
    )

    # ---------------------------------------------------
    # compute image2d RSS from image in detector, method1
    # ---------------------------------------------------
    image2d_rss_method1 = compute_image2d_rss_from_detector_method1(
        image2d_detector_method0=image2d_detector_method0,
        naxis1_detector=naxis1_detector,
        naxis1_ifu=naxis1_ifu,
        nslices=nslices,
        dict_ifu2detector=dict_ifu2detector,
        wcs3d=wcs3d,
        parallel_computation=parallel_computation,
        logger=logger,
    )

    # save FITS file
    save_image2d_rss(
        wcs3d=wcs3d,
        header_keys=header_keys,
        image2d_rss=image2d_rss_method1,
        method=1,
        prefix_intermediate_fits=prefix_intermediate_fits,
        bitpix=-32,
        logger=logger,
        output_dir=output_dir,
    )

    # ------------------------------------
    # compute image3d IFU from RSS method1
    # ------------------------------------
    image3d_ifu_method1 = compute_image3d_ifu_from_rss_method1(
        image2d_rss_method1=image2d_rss_method1,
        naxis1_detector=naxis1_detector,
        naxis2_ifu=naxis2_ifu,
        naxis1_ifu=naxis1_ifu,
        nslices=nslices,
        logger=logger,
    )

    # save FITS file
    if len(prefix_intermediate_fits) > 0:
        hdu = fits.PrimaryHDU(image3d_ifu_method1.astype(np.float32))
        pos0 = len(hdu.header) - 1
        hdu.header.extend(wcs_to_header_using_cd_keywords(wcs3d), update=True)
        hdu.header.update(header_keys)
        hdu.header.insert(pos0, ("COMMENT", "FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        hdu.header.insert(
            pos0 + 1, ("COMMENT", "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H")
        )
        hdul = fits.HDUList([hdu])
        outfile = f"{prefix_intermediate_fits}_ifu_3D_method1.fits"
        logger.info(f"Saving file: {outfile}")
        hdul.writeto(f"{Path(output_dir) / outfile}", overwrite=True)
