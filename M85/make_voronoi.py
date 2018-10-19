# -*- coding: utf-8 -*-
""" 

Created on 18/10/18

Author : Carlos Eduardo Barbosa

Apply Voronoi tesselation to WIFIS datacubes.

"""

from __future__ import print_function, division

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from vorbin.voronoi_2d_binning import voronoi_2d_binning

import context

def make_voronoi(targetSN, redo=False):
    """ Determination of SNR for each spaxel. """
    datacube = os.path.join(context.data_dir, "M85_combined_cube_2.fits")
    errcube = os.path.join(context.data_dir, "M85_unc_cube.fits")
    output = os.path.join(context.data_dir, "voronoi_sn{}.fits".format(
        targetSN))
    if os.path.exists(output) and not redo:
        return
    data = fits.getdata(datacube)
    err = fits.getdata(errcube)
    signal = np.nanmedian(data, axis=0)
    sn3D = data / err
    sn = np.nanmedian(sn3D, axis=(0,1))
    noise = signal / sn
    zdim, ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim))
    # Selecting only saxels where the percentage of nans is low
    nnans = np.isnan(data).sum(axis=0)
    mask = np.where(nnans < 80, 0, 1)
    # Preparing arrays for Voronoi binning
    idx = mask==0
    xpix = xx[idx] + 1
    ypix = yy[idx] + 1
    signal = signal[idx]
    noise = noise[idx]
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, \
    scale = voronoi_2d_binning(xpix, ypix, signal, noise, targetSN, plot=False,
                               quiet=True, pixelsize=1, cvt=True)
    binNum += 1
    voronoi = np.zeros_like(xx) * np.nan
    voronoi[idx] = binNum
    vorHDU = fits.PrimaryHDU(voronoi)
    vorHDU.header["EXTNAME"] = "VORONOI2D"
    table = Table([np.unique(binNum), xNode, yNode, xBar, yBar, sn, nPixels,
                             scale],
                  names=["binnum", "xpix", "ypix", "xpix_bar", "ypix_bar",
                         "snr", "npix", "scale"])
    tabHDU = fits.BinTableHDU(table)
    tabHDU.header["EXTNAME"] = "TABLE"
    hdulist = fits.HDUList([vorHDU, tabHDU])
    hdulist.writeto(output, overwrite=True)

if __name__ == "__main__":
    targetSN = 80
    make_voronoi(targetSN, redo=True)
