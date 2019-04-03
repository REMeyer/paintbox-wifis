# -*- coding: utf-8 -*-
""" 

Created on 23/10/18

Author : Carlos Eduardo Barbosa

Produces maps of ATLAS3D cubes for comparison.
"""
from __future__ import print_function, division

import os

from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import context

if __name__ == "__main__":
    table_file = os.path.join(context.data_dir, "PXF_bin_MS_NGC4382_r5_idl.fits")
    data = Table.read(table_file)
    fields = ["VPXF", 'SPXF', 'H3PXF', 'H4PXF', "VPXF_VS", "SPXF_VS"]
    labels = ["V (km/s)", "$\sigma$ (km/s)", "$h_3$", "$h_4$", "V (km/s)", "$\sigma$ (km/s)"]
    outdir = os.path.join(context.data_dir, "plots", "atlas3d")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i, field in enumerate(fields):
        fig = plt.figure(figsize=(2.7, 4))
        ax = fig.add_subplot(111, aspect='equal')
        ax.minorticks_on()
        im = ax.scatter(data["YS"], data["XS"], c=data[field],
                        cmap="Spectral_r")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-24, 24)
        ax.set_xlabel("$\Delta$X (arcsec)")
        ax.set_ylabel("$\Delta$Y (arcsec)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(labels[i])
        plt.subplots_adjust(left=0.11, top=0.98, bottom=0.12, right=0.88)
        plt.savefig(os.path.join(outdir, "{}.png".format(field)), dpi=300)