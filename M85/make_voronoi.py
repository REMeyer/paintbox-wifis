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

import context
from misc import snr

def make_voronoi(data, targetSN, redo=False):
    """ Determination of SNR for each spaxel.

    Input parameters
    ----------------
    data : np.array
        Science data cube

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
    output = "voronoi_sn{}.fits".format(targetSN)
    if os.path.exists(output) and not redo:
        return
    signal, noise, sn = snr(data)
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

def combine_spectra(data, error, header, targetSN, redo=False):
    """ Produces the combined spectra for a given binning file.

    Input Parameters
    ----------------
    data: np.array
        Science data cube data

    error: np.array
       Uncertainty data cube data.

    targetSN : float
        Value of the SNR ratio used in the tesselation

    redo : bool
        Redo combination in case the output spec already exists.

    Output
    ------
    All the combined spectra are written in FITS table format and stored in
    a folder called spec_sn{targetSN}.

    """
    outdir = os.path.join(os.getcwd(), "spec1d_sn{}".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    wave = ((np.arange(header['NAXIS3']) + 1
             - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']) * u.m
    vordata = fits.getdata("voronoi_sn{}.fits".format(targetSN), 0)
    bins = np.unique(vordata[~np.isnan(vordata)])
    filelist = []
    for j, bin in enumerate(bins):
        idy, idx = np.where(vordata == bin)
        ncombine = len(idx)
        print("Bin {0} / {1} (ncombine={2})".format(j + 1, bins.size, ncombine))
        output = os.path.join(outdir, "sn{}_{:04d}.fits".format(targetSN,
                                                                int(bin)))
        if os.path.exists(output) and not redo:
            continue
        specs = data[:,idy,idx]
        errors = error[:,idy,idx]
        errs = np.sqrt(np.nansum(errors**2, axis=1))
        combined = np.nanmean(specs, axis=1) * ncombine
        combined[np.isnan(combined)] = 0
        mask = np.where(combined==0, 0., 1.)
        table = Table([wave.to("micrometer").data, combined, errs, mask],
                      names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
        table.write(output, overwrite=True)
        filelist.append(output)
    with open("filelist_sn{}.txt".format(targetSN), "w") as f:
        f.write("\n".join(filelist))
    return

if __name__ == "__main__":
    os.chdir(context.data_dir)
    # Input
    targetSN = 40
    datacube = "M85_combined_cube_2.fits"
    errcube = "M85_unc_cube.fits"
    # Reading data
    data = fits.getdata(datacube)
    errdata = fits.getdata(errcube)
    errdata = errdata.sum(axis=0)# Removing extra dimension
    # Normalizing by the exposure time
    header = fits.getheader(datacube)
    exptime = header["OBJTIME"]
    data /=  exptime
    errdata /= exptime
    # Processing data
    make_voronoi(data, targetSN, redo=False)
    combine_spectra(data, errdata, header, targetSN, redo=True)