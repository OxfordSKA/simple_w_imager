# coding: utf-8
""""""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import math
import numpy as np
from simple_w_imager.kernels import (exp_taper, generate_w_kernel)
from simple_w_imager.utils import (plot_image, plot_cimage, plot_line,
                                   plot_semilogy)


def main():
    """."""
    im_size = 2**14
    fov = 4.0
    w = 10000
    num_w = 16

    w_max = 15000
    w_min = 0
    w_mid = 0.5 * (w_min + w_max)
    w_rms = w_mid * 0.5
    max_uvw = 1.05 * w_max * (w_rms / w_mid)
    w_scale = (num_w - 1)**2 / max_uvw

    kernels = []
    for i in range(num_w):
        w = i**2 / w_scale
        c_uv, support = generate_w_kernel(w, im_size, fov, max_size=512,
                                          over_sample=4, padded=True,
                                          cut_level=1e-3)
        print(i, w, support)
        kernels.append(dict(kernel=c_uv, support=support))
        plot_cimage(kernels[i]['kernel'],
                    title='w = {}, support = {}'.format(w, support))



if __name__ == '__main__':
    main()
