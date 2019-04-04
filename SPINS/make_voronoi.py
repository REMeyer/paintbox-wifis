# -*- coding: utf-8 -*-
""" 

Created on 18/10/18

Author : Carlos Eduardo Barbosa

Apply Voronoi tesselation to WIFIS datacubes.

"""

from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from vorbin.voronoi_2d_binning import voronoi_2d_binning

import misc

def make_voronoi(datacube, targetSN, output, redo=False):
    """ Determination of SNR for each spaxel.

    Input parameters
    ----------------
    data : str
        Name of the science data cube.

    targetSN: float
        Value of the goal SNR.

    redo: bool
        Redo tesselation in case the output already exists.

    Output
    ------
    The program generates a multi-extension FITS file containing two extensions:
        1: Voronoi tesselation map
        2: Table with information about the tesselation
    """
    data = fits.getdata(datacube)
    if os.path.exists(output) and not redo:
        return
    signal, noise, sn = misc.snr(data)
    zdim, ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim))
    # Selecting only spaxels where the percentage of nans is low
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

def combine_spectra(datacube, vorfile, outdir, redo=False, error=None):
    """ Produces the combined spectra for a given binning file.

    Input Parameters
    ----------------
    data: str
        Science data cube data

    vorfile: str
        Name of the file containing the Voronoi binning information

    outdir: str
        Directory where the spectra will be saved.

    error: np.array
       Uncertainty data cube data.

    redo : bool
        Redo combination in case the output spec already exists.


    """
    data = fits.getdata(datacube)
    header = fits.getheader(datacube)
    wave = ((np.arange(header['NAXIS3']) + 1
             - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']) * u.m
    vordata = fits.getdata(vorfile, 0)
    bins = np.unique(vordata[~np.isnan(vordata)])
    print("Combining spectra for Voronoi binning...")
    for j, bin in enumerate(bins):
        idy, idx = np.where(vordata == bin)
        ncombine = len(idx)
        output = os.path.join(outdir, "spec{:04d}.fits".format(int(bin)))
        if os.path.exists(output) and not redo:
            continue
        print("Bin {0} / {1} (ncombine={2})".format(j + 1, bins.size, ncombine))
        specs = data[:,idy,idx]
        combined = np.nanmean(specs, axis=1) * ncombine
        combined[np.isnan(combined)] = 0
        mask = np.where(combined == 0, 0., 1.)
        if error is not None:
            errors = error[:,idy,idx]
            errs = np.sqrt(np.nansum(errors**2, axis=1))
        else:
            errs = np.zeros_like(combined)
        table = Table([wave.to("micrometer").data, combined, errs, mask],
                      names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
        table.write(output, overwrite=True)
    print("Done!")
    return