# coding: utf-8
"""Module for utility functions."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def fov_to_cell_size(fov, im_size):
    """Evaluate image pixel size for given FoV and number of pixels.

    Args:
        fov (float): Field of view, in degrees
        im_size (int): Image size (1D) in pixels.

    Returns:
        double: pixel size, in arcsec.
    """
    r_max = math.sin(math.radians(fov) / 2)
    inc = r_max / (0.5 * im_size)
    return math.degrees(math.asin(inc)) * 3600


def cell_size_to_fov(cell_size, im_size):
    """Obtain image fov from cell size and image size."""
    inc = math.sin(math.radians(cell_size / 3600))
    r_max = inc * (0.5 * im_size)
    return math.degrees(2.0 * math.asin(r_max))


def plot_image(image, title=None, cbar_label=None, filename=None, extent=None):
    if extent is None:
        size = image.shape[0]
        extent = [-size // 2 - 0.5, size // 2 - 1 + 0.5,
                  -size // 2 - 0.5, size // 2 - 1 + 0.5]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.03)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    ticks = np.linspace(image.min(), image.max(), 5)
    cbar.set_ticks(ticks, update_ticks=True)
    ax.grid()
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    if filename:
        fig.savefig(filename)
    else:
        plt.show()


def plot_cimage(image, title=None, cbar_label=None, filename=None, extent=None):
    if extent is None:
        size = image.shape[0]
        extent = [-size // 2 - 0.5, size // 2 - 1 + 0.5,
                  -size // 2 - 0.5, size // 2 - 1 + 0.5]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92,
                        wspace=0.4, hspace=None)
    for i, ax in enumerate(axes):
        if i == 0:
            data = np.real(image)
            title_ = 'Real ( %s )' % title
        else:
            data = np.imag(image)
            title_ = 'Imag ( %s )' % title
        im = ax.imshow(data, interpolation='nearest', extent=extent,
                       origin='lower', cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cbar = ax.figure.colorbar(im, cax=cax)
        ticks = np.linspace(data.min(), data.max(), 5)
        cbar.set_ticks(ticks, update_ticks=True)
        if cbar_label:
            cbar.set_label(cbar_label)
        if title:
            ax.set_title(title_)
    if filename:
        fig.savefig(filename)
    else:
        plt.show()


def plot_line(y, x=None, title=None, filename=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    if x is None:
        x = np.arange(-y.shape[0]//2, y.shape[0]//2)
    ax.plot(x, y, 'b.--')
    ax.plot([x[0], x[-1]], [0.5, 0.5], 'r--')
    ax.grid()
    if title:
        ax.set_title(title)
    if filename:
        fig.savefig(filename)
    else:
        plt.show()


def plot_semilogy(y, x=None, title=None, filename=None, y_lim=None, x1=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    if x is None:
        x = np.arange(-y.shape[0]//2, y.shape[0]//2)
    ax.semilogy(x, y, 'b.--')
    ax.semilogy([x[0], x[-1]], [0.5, 0.5], 'r--')

    if y_lim:
        ax.set_ylim(y_lim)
    if x1:
        ax.plot([x1, x1], ax.get_ylim(), '--', color='0.5')
        ax.plot([-x1, -x1], ax.get_ylim(), '--', color='0.5')
    ax.grid()
    if title:
        ax.set_title(title)
    if filename:
        fig.savefig(filename)
    else:
        plt.show()


def plot_cimage_2(cimage, title=None, cbar_label=None, filename=None,
                 extent=None):
    plot_image(np.real(cimage), title='Real ( %s )' % title,
               cbar_label=cbar_label,
               filename = filename,
               extent = extent)
    plot_image(np.imag(cimage), title='Imag ( %s )' % title,
               cbar_label=cbar_label,
               filename = filename,
               extent = extent)