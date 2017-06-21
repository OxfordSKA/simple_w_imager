# coding: utf-8
"""
Main script used to generate plots for Fred's many core talk
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import math
import numpy as np
from simple_w_imager.kernels import (generate_w_kernel, exp_taper,
                                     w_kernel_lm_padded, w_kernel_lm_extended,
                                     kernel_dims)
from simple_w_imager.utils import (plot_image, plot_cimage, plot_line,
                                   plot_semilogy)


def extent_lm(fov_deg, size):
    """."""
    c = size // 2
    extent = np.array([c + 0.5, -c + 0.5, -c - 0.5, c - 0.5])
    cell_rad = math.asin(2.0 * math.sin(0.5 * math.radians(fov_deg)) / size)
    cellsize_lm = math.sin(cell_rad)
    extent *= cellsize_lm
    return extent


def extent_uv(fov_deg, size):
    """."""
    c = size // 2
    cell_rad = math.asin(2.0 * math.sin(0.5 * math.radians(fov_deg)) / size)
    cell_uv = 1.0 / (size * cell_rad)
    extent = np.array([c + 0.5, -c + 0.5, -c - 0.5, c - 0.5])
    extent *= cell_uv
    return extent


def plot_kernels(w):
    """."""
    im_size = 2048
    fov = 4
    over_sample = 4
    max_size = 8192
    fwhm_fov_fraction = 0.4
    padded = True
    cut_level = 1e-3

    lm_inc, conv_size, inner = kernel_dims(fov, im_size, max_size, over_sample)
    cell_rad = math.asin(2.0 * math.sin(0.5 * math.radians(fov)) / im_size)
    lm_max = math.sin(cell_rad) * (im_size // 2)
    print(cell_rad, lm_max)
    print(lm_inc, conv_size, inner)

    if padded:
        x_inner = np.arange(-inner // 2, inner // 2 + 1) * lm_inc
        l_inner, m_inner = np.meshgrid(-x_inner, x_inner)
        r2_inner = l_inner**2 + m_inner**2
        w_lm = w_kernel_lm_padded(w, im_size, fov, over_sample, max_size)
    else:
        x_outer = np.arange(-conv_size // 2, conv_size // 2) * lm_inc
        l_outer, m_outer = np.meshgrid(-x_outer, x_outer)
        r2_outer = l_outer**2 + m_outer**2
        w_lm = w_kernel_lm_extended(w, im_size, fov, over_sample, max_size)

    fwhm = (w_lm.shape[0] // over_sample) * fwhm_fov_fraction
    t_lm = exp_taper(w_lm.shape[0], fwhm=fwhm)
    # c_lm = t_lm
    # c_lm = w_lm
    c_lm = t_lm * w_lm


    print(c_lm.shape, im_size, inner, conv_size)
    c = conv_size // 2
    e1 = c - inner // 2 - 100
    e2 = c + inner // 2 + 1 + 100
    c_lm_plot = c_lm[e1:e2, e1:e2]
    # c_lm_plot = c_lm
    extent = extent_lm(fov, im_size)
    # filename = 'c_lm_w{}.png'.format(w)
    # filename = None
    # plot_cimage(c_lm_plot,
    #             filename=filename, extent=extent, figsize=(12, 10),
    #             title='w = {}'.format(w))
    filename = 'r_lm_w{}.png'.format(w)
    print(extent)
    # filename = None
    plot_image(np.real(c_lm_plot),
               filename=filename, extent=extent, figsize=(6, 6),
               title='w = {}'.format(w), xlabel='l', ylabel='m')

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
    w_uv = c_uv[c0:c1, c0:c1]

    cell_uv = 1.0 / (im_size * cell_rad)
    cell_uv /= over_sample
    uv_max = w_uv.shape[0] // 2 * cell_uv
    print(uv_max)
    filename = 'r_uv_w{}.png'.format(w)
    extent = extent_uv(fov * over_sample, w_uv.shape[0])
    print(extent)
    plot_image(np.real(w_uv),
               filename=filename, extent=extent, figsize=(6, 6),
               title='w = {}'.format(w), xlabel='uu (lambda)',
               ylabel='vv (lambda)', xlim=[-300, 300], ylim=[-300, 300])
    #
    #
    # plot_cimage(w_uv)


def main():
    """."""
    plot_kernels(w=0)
    plot_kernels(w=5000)
    plot_kernels(w=10000)
    plot_kernels(w=15000)

if __name__ == '__main__':
    main()
