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
        float: pixel size, in arcsec.
    """
    r_max = math.sin(math.radians(fov) / 2)
    inc = r_max / (0.5 * im_size)
    return math.degrees(math.asin(inc)) * 3600


def cell_size_to_fov(cell_size, im_size):
    """Obtain image fov from cell size and image size.

    Args:
        cell_size (float): Cell size, in arcseconds.
        im_size (int): Image size, in pixels.

    Returns:
        float, The image FoV, in degrees.
    """
    inc = math.sin(math.radians(cell_size / 3600))
    r_max = inc * (0.5 * im_size)
    return math.degrees(2.0 * math.asin(r_max))


def plot_image(image, title=None, cbar_label=None, filename=None, extent=None,
               figsize=(8, 4), xlabel=None, ylabel=None, xlim=None,
               ylim=None):
    """Utility function to plot a image.

    Args:
        image (numpy.array, complex): Complex image / 2d-array to be plotted.
        title (str, optional): Plot title.
        cbar_label (str, optional): Color bar label
        filename (str, optional): If specified, save the plot to this file
                                  instead of displaying the plot.
        extent (list, optional): If specified the extent of the plot axis labels
                                 [x_min, x_max, y_min, y_max]
    """
    if extent is None:
        size = image.shape[0]
        extent = [-size // 2 - 0.5, size // 2 - 1 + 0.5,
                  -size // 2 - 0.5, size // 2 - 1 + 0.5]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.03)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    ticks = np.linspace(image.min(), image.max(), 5)
    cbar.set_ticks(ticks, update_ticks=True)
    # ax.grid()
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if filename:
        fig.savefig(filename, dpi=200, transparent=True)
    else:
        plt.show()


def plot_cimage(image, title=None, cbar_label=None, filename=None, extent=None,
                figsize=(8, 4)):
    """Utility function to plot a complex image.

    Args:
        image (numpy.array, complex): Complex image / 2d-array to be plotted.
        title (str, optional): Plot title.
        cbar_label (str, optional): Color bar label
        filename (str, optional): If specified, save the plot to this file
                                  instead of displaying the plot.
        extent (list, optional): If specified the extent of the plot axis labels
                                 [x_min, x_max, y_min, y_max]
    """
    if extent is None:
        size = image.shape[0]
        extent = [-size // 2 - 0.5, size // 2 - 1 + 0.5,
                  -size // 2 - 0.5, size // 2 - 1 + 0.5]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
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
    """Utility function to generate a line plot.

        Args:
            y (numpy.array, 1d): y data array to be plotted.
            x (numpy.array, optional): If specified, the x data. Otherwise
                                       The x data values are assumes to be an
                                       integer range from 0 to y.size.
            title (str, optional): Plot title.
            filename (str, optional): If specified, save the plot to this file
                                      instead of displaying the plot.
    """
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
    """Utility function to generate a line plot with a log y axis.

    Args:
        y (numpy.array, 1d): y data array to be plotted.
        x (numpy.array, optional): If specified, the x data. Otherwise
                                   The x data values are assumes to be an
                                   integer range from 0 to y.size.
        title (str, optional): Plot title.
        filename (str, optional): If specified, save the plot to this file
                                  instead of displaying the plot.
        y_lim (list, optional): y range of the axis [y_min, y_max]
        x1 (float, optional): If specified plot a vertical guide line at this
                              x position.
    """
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
