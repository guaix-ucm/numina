import sys

import numpy
import pytest

from ..findpeaks1D import findPeaks_spectrum, refinePeaks_spectrum

from ..peakdet import generate_kernel
from ..peakdet import find_peaks_index1
from ..peakdet import find_peaks_index2
from ..peakdet import refine_peaks1, refine_peaks2, refine_peaks3
from .._kernels import kernel_peak_function

@pytest.mark.skipif(sys.version_info < (3,0),
                    reason="requires python3")
def test_pycapsule():
    m = kernel_peak_function(20)
    # This seems to be the only way to check that something is a PyCapsule
    sm = type(m).__name__
    assert sm == 'PyCapsule'


@pytest.mark.skipif(sys.version_info >= (3,0),
                    reason="requires python2")
def test_pycobject():
    m = kernel_peak_function(20)
    # This seems to be the only way to check that something is a PyCapsule
    sm = type(m).__name__
    assert sm == 'PyCObject'

@pytest.mark.parametrize("window", [3, 5, 7, 9])
def test_generate_kernel(benchmark, window):
    result = benchmark(generate_kernel, window)

    assert result.shape == (3, window)


@pytest.fixture(scope="module")
def spectrum():
    yl = [
        1.929785E-16,
        1.947651E-16,
        1.931419E-16,
        1.720405E-16,
        1.631675E-16,
        1.412115E-16,
        2.630947E-16,
        5.293785E-16,
        5.981209E-16,
        5.857859E-16,
        4.436979E-16,
        1.942074E-16,
        1.407296E-16,
        1.487431E-16,
        1.466912E-16,
        1.179581E-16,
        1.492219E-16,
        1.694441E-16,
        1.418203E-16,
        1.492211E-16,
        1.367023E-16,
        1.312262E-16,
        1.546229E-16,
        1.502321E-16,
        1.417931E-16,
        1.472845E-16,
        1.496578E-16,
        1.635401E-16,
        1.688833E-16,
        1.717370E-16,
        1.857556E-16,
        1.721150E-16,
        1.827260E-16,
        1.778807E-16,
        1.519536E-16,
        1.998407E-16,
        2.377358E-16,
        2.551754E-16,
        3.063046E-16,
        3.547860E-16,
        3.767656E-16,
        3.839578E-16,
        2.990524E-16,
        2.179336E-16,
        1.979480E-16,
        2.226581E-16,
        1.998271E-16,
        1.859107E-16,
        1.801761E-16,
        1.624808E-16,
        1.567423E-16,
        1.712928E-16,
        1.840137E-16,
        1.740313E-16,
        3.055553E-16,
        4.885267E-16,
        5.264164E-16,
        4.586895E-16,
        3.249551E-16,
        2.160603E-16,
        1.816723E-16,
        1.699870E-16,
        1.808596E-16,
        1.825298E-16,
        1.687753E-16,
        1.628289E-16,
        1.652891E-16,
        1.798384E-16,
        1.826757E-16,
        1.948041E-16,
        1.925638E-16,
        1.858497E-16,
        1.855130E-16,
        1.851734E-16,
        1.960223E-16,
        2.007465E-16,
        1.918781E-16,
        1.683171E-16,
        2.168162E-16,
        2.496364E-16,
        2.696567E-16,
        2.664617E-16,
        2.430849E-16,
        2.109349E-16,
        1.823294E-16,
        2.622979E-16,
        8.927012E-16,
        1.476848E-15,
        1.701959E-15,
        1.609862E-15,
        8.395067E-16,
        2.951189E-16,
        2.051973E-16,
        1.873397E-16,
        1.926380E-16,
        1.874719E-16,
        1.868016E-16,
        1.823945E-16,
        1.661913E-16,
        1.706098E-16
    ]
    return numpy.array(yl)


def test_peak_finding_base(benchmark, spectrum):
    peakin = [8, 56, 88]
    result = benchmark(findPeaks_spectrum, spectrum, 5, 0.5e-15)
    assert numpy.allclose(peakin, result)


def test_peak_finding_v1(benchmark, spectrum):
    # Reference
    peakin = [8, 56, 88]
    result = benchmark(find_peaks_index1, spectrum, threshold=0.5e-15, window=5)
    assert numpy.allclose(peakin, result)


def test_peak_finding_v2(benchmark, spectrum):
    # Reference
    peakin = [8, 56, 88]
    result = benchmark(find_peaks_index2, spectrum, threshold=0.5e-15, window=5)
    assert numpy.allclose(peakin, result)


def test_peak_refine_original(benchmark, spectrum):
    peakin = findPeaks_spectrum(spectrum, 5, 0.5e-15)
    peakpos = benchmark(refinePeaks_spectrum, spectrum, peakin, 5, method=1)

    assert peakpos is not None


def test_peak_refine_loop(benchmark, spectrum):

    peakin = findPeaks_spectrum(spectrum, 5, 0.5e-15)
    peakpos = benchmark(refine_peaks1, spectrum, peakin, 5)

    assert peakpos is not None


def test_peak_refine_no_loop(benchmark, spectrum):

    peakin = findPeaks_spectrum(spectrum, 5, 0.5e-15)
    peakpos = benchmark(refine_peaks2, spectrum, peakin, 5)

    assert peakpos is not None


def test_peak_refine_no_loop_compW(benchmark, spectrum):

    peakin = findPeaks_spectrum(spectrum, 5, 0.5e-15)
    peakpos = benchmark(refine_peaks3, spectrum, peakin, 5)

    assert peakpos is not None
