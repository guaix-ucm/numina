"""The figures of compute_crmasks are saved in files without pyplot"""

import matplotlib.pyplot as plt
import numpy as np

from numina.array.crmasks.compute_crmasks import compute_crmasks

NAXIS = 50


def make_images():
    """Three images with one cosmic ray each and a cosmic ray in the same pixels of two images"""
    rng = np.random.default_rng(1)
    images = []
    for idx in range(3):
        image = rng.normal(1000.0, 10.0, size=(NAXIS, NAXIS)).astype(np.float32)
        image[10 + 5 * idx, 20 + 3 * idx] += 5000
        if idx < 2:
            image[30:32, 30:32] += 8000
        images.append(image)
    return images


def test_compute_crmasks_figures(tmp_path):
    hdul = compute_crmasks(
        make_images(),
        output_dir=tmp_path,
        gain=1.0,
        rnoise=10.0,
        bias=0.0,
        crmethod="mm_lacosmic",
        flux_factor="auto",
        apply_flux_factor_to="simulated",
        interactive=False,
        debug=True,
        la_sigclip=[5.0, 3.0],
        la_sigfrac=[0.3, 0.3],
        la_objlim=[5.0, 5.0],
        la_satlevel=65535.0,
        la_niter=4,
        la_sepmed=True,
        la_cleantype="meanmask",
        la_fsmode="convolve",
        la_psfmodel="gaussxy",
        la_psffwhm_x=2.5,
        la_psffwhm_y=2.5,
        la_psfsize=11,
        la_psfbeta=4.765,
        la_padwidth=10,
        mm_cr_coincidences=2,
        mm_photon_distribution="poisson",
        mm_nbinom_shape=1000,
        mm_synthetic="median",
        mm_hist2d_min_neighbors=0,
        mm_ydiag_max=0,
        mm_dilation=0,
        mm_crosscorr_region=[1, NAXIS, 1, NAXIS],
        mm_boundary_fit="spline",
        mm_knots_splfit=3,
        mm_nsimulations=1,
        mm_niter_boundary_extension=3,
        mm_weight_boundary_extension=10,
        mm_threshold_rnoise=0.0,
        mm_minimum_max2d_rnoise=5.0,
        mm_seed=1234,
    )
    assert hdul["MEDIANCR"].data[30:32, 30:32].all()
    # compute_flux_factor, cross-correlation, display_hist2d,
    # diagnostic_plot and display_detected_cr
    for filename in [
        "histogram_before_flux_factor.png",
        "image_number_at_median_position.png",
        "flux_factor1.png",
        "xyoffset_crosscorr_1.png",
        "diagnostic_histogram2d.png",
        "diagnostic_mediancr.png",
        "mediancr_identified_only2.pdf",
    ]:
        assert (tmp_path / filename).stat().st_size > 0
    assert plt.get_fignums() == []
