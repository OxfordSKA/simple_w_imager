# coding: utf-8
"""Module for generating gridding kernels."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import math
import numpy as np
from .utils import plot_image, plot_cimage


def kernel_dims(fov, image_size, max_size, over_sample):
    """Return key kernel dimensions.
    
    Args:
        fov (float): Image field-of-view, in degrees.
        image_size (int): Image size, in pixels.
        max_size (int): max size of the kernel, in pixels.
        over_sample (int): Kernel over-sample, points per grid cell.
    
    Returns:
        tuple: (kernel lm cellsize, outer kernel size, inner kernel size)
    """
    assert max_size % 2 == 0
    size = min(image_size * over_sample, max_size)  # Size of the kernel.
    inner = size // over_sample
    # If padding the grid, increase the fov here to match.
    # eg. (2 * lm_max * padding=1.05)
    lm_max = math.sin(0.5 * np.radians(fov))
    lm_inc = (2 * lm_max) / inner
    # print(lm_inc, size, inner, (2 * lm_max) / image_size, image_size/inner)
    return lm_inc, size, inner


def w_kernel_lm_extended(w, image_size, fov, over_sample=4, max_size=2560):
    """Generate image plane w-kernel for the entire over-sampled fov, w(l,m)

    Args:
        w (float): w-coordinate value, in wavelengths.
        image_size (int): Image size, in pixels.
        fov (float): Image field of view, in degrees.
        over_sample (int): kernel over-sample factor.
        max_size (int): max kernel size, in pixels.

    Returns:
        numpy.array containing the image plane w phase screen w(l, m)
    """
    lm_inc, conv_size, _ = kernel_dims(fov, image_size, max_size, over_sample)

    # Kernel coordinate grid in l,m space.
    x_outer = np.arange(-conv_size // 2, conv_size // 2) * lm_inc
    l_outer, m_outer = np.meshgrid(-x_outer, x_outer)
    r2_outer = l_outer**2 + m_outer**2

    # Image plane w-projection kernel.
    w_lm = np.exp(-1.j * 2. * w * np.pi * (np.sqrt(1. - r2_outer) - 1.))
    return w_lm


def w_kernel_lm_padded(w, image_size, fov, over_sample=4, max_size=2560):
    """Generate image plane w-kernel zero-padded by the over-sample factor.

    Args:
        w (float): w-coordinate value, in wavelengths.
        image_size (int): Image size, in pixels.
        fov (float): Image field of view, in degrees.
        over_sample (int): kernel over-sample factor.
        max_size (int): max kernel size, in pixels.
        
    Returns:
        numpy.array containing the image plane w phase screen w(l, m)
    """
    lm_inc, conv_size, inner = kernel_dims(fov, image_size, max_size,
                                           over_sample)

    # Kernel coordinate grid in l,m space.
    x_inner = np.arange(-inner // 2, inner // 2 + 1) * lm_inc
    l_inner, m_inner = np.meshgrid(-x_inner, x_inner)
    r2_inner = l_inner**2 + m_inner**2

    w_lm = np.zeros((conv_size, conv_size), 'c16')
    c = conv_size // 2
    e1 = c - inner // 2
    e2 = c + inner // 2 + 1
    w_lm[e1:e2, e1:e2] = np.exp(-1.j * 2. * w * np.pi * (np.sqrt(1. - r2_inner) - 1.))

    return w_lm


def exp_taper(size, fwhm):
    """Gaussian taper function.
    
    Args:
        size (int): Size of the taper function, in pixels.
        fwhm (float): FHWM of the taper, in pixels.
    
    Returns:
        numpy.array containing the taper fuction.
    """
    x = np.arange(-size / 2, size / 2, dtype='f8')
    x, y = np.meshgrid(x, -x)
    sigma = fwhm / (2 * (2 * math.log(2))**0.5)
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


def generate_w_kernel(w, image_size, fov, over_sample=4, max_size=2560,
                      fwhm_fov_fraction=0.4, cut_level=1e-3, padded=False):
    """Generate uv plane w-projection kernel.
    
    Args:
        w (float): w-coordinate value for the kernel, in wavelengths
        image_size (int): Image size, in pixels
        fov (float): Image FoV, in degrees.
        over_sample (int): Kernel over-sample. ie. number of points per grid 
                           cell.
        max_size (int): Maximum size of the kernel, in pixels.
        fwhm_fov_fraction (float): Fraction of the FoV used for the FWHM of the
                                   Gaussian taper.
        cut_level (float): Level at which the uv plane kernel is clipped.
        padded (bool): If true generate a zero padded kernel, if false
                       evaluate the kernel all the way out to the edge of the
                       extended FoV.
    
    Returns:
        tuple (numpy.array, int)
          numpy.array containing the u,v plane tapered w-kernel w(u, v)
          support size of the kernel.
    """
    # Evaluate w phase screen.
    if padded:
        w_lm = w_kernel_lm_padded(w, image_size, fov, over_sample, max_size)
    else:
        w_lm = w_kernel_lm_extended(w, image_size, fov, over_sample, max_size)

    # plot_cimage(w_lm)
    # plot_image(np.abs(w_lm))

    # Evaluate taper function
    fwhm = (w_lm.shape[0] // over_sample) * fwhm_fov_fraction
    t_lm = exp_taper(w_lm.shape[0], fwhm=fwhm)

    # Apply taper
    c_lm = t_lm * w_lm

    # plot_cimage(c_lm)

    # FFT to uv plane and normalise
    c_uv = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(c_lm)))
    c_uv /= np.max(np.abs(c_uv))

    # Work out kernel clipping indices for specified cut level.
    size = c_uv.shape[0]
    centre = size // 2
    cut_idx0 = np.argmax(np.abs(c_uv[centre, :]) > cut_level)
    cut_idx1 = centre + (centre - cut_idx0)
    w_uv_width = cut_idx1 - cut_idx0 - 1
    w_uv_width = math.ceil(float(w_uv_width) / over_sample)
    support = (w_uv_width - 1) // 2
    c0 = centre - support * over_sample - over_sample / 2
    c1 = centre + support * over_sample + over_sample / 2
    c0, c1 = int(c0), int(c1)

    return c_uv[c0:c1, c0:c1], support
