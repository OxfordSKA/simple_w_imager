# coding: utf-8
"""Module for simple gridding functions."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
from .utils import fov_to_cell_size
import numpy as np


def w_proj_gridder(fov, im_size, uu, vv, ww, amp, kernels, padding=1.05):
    """
    
    Args:
            
    
    """
    grid = np.zeros((im_size, im_size), 'c16')
    cell_size = fov_to_cell_size(fov, im_size)

    num_vis = uu.shape[0]
    x = -uu / cell_size
    y = vv / cell_size
    xg = int(round(x))
    yg = int(round(y))
    x_offset = xg - x
    y_offset = yg - y

