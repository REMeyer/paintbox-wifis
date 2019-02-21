# -*- coding: utf-8 -*-
""" 

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Preprocessing of the data before the use of molecfit for the telluric
correction.

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from photutils import DAOStarFinder, aperture_photometry, CircularAperture
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
import matplotlib.pyplot as plt

import context
from spectres import spectres

def std_phot(cube, img, output, r=30, redo=False):
    """Determination of aperture and extraction of 1D spectrum. """
    if os.path.exists(output) and not redo:
        return
    ############################################################################
    # Defining the aperture with maximum S/N using the image as a reference
    data = fits.getdata(img)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    daofind = DAOStarFinder(fwhm=2.1, threshold=10.*std)
    star= daofind(data - median)
    positions = [(star["xcentroid"][0], star["ycentroid"][0])]
    # Separating region inside aperture from the outside to determine sky
    ydim, xdim = data.shape
    X, Y = np.meshgrid(np.arange(xdim), np.arange(ydim))
    R = np.sqrt((X - star["xcentroid"])**2 + (Y - star["ycentroid"])**2)
    ############################################################################
    # Extracting stellar spectrum from the datacube
    cube = fits.getdata(cube)
    idx = np.where(R <= r)
    npix = len(idx[0])
    specs = cube[:, idx[0], idx[1]]
    hdr = fits.getheader(stdcube)
    wave = ((np.arange(hdr['NAXIS3']) + 1 - hdr['CRPIX3']) * hdr['CDELT3'] + \
           hdr['CRVAL3']) * u.m
    spec1D = np.nansum(specs, axis=1)
    # Extracting background spectrum
    idxb = np.where(R > r)
    bkg = np.nanmedian(cube[:, idxb[0], idxb[1]], axis=1) * npix
    # Sky subtraction
    spec1D -= bkg
    # Preparing output table
    err = np.zeros_like(spec1D) # necessary for molecfit table
    mask = np.ones_like(spec1D)
    mask[np.isnan(spec1D)] = 0.
    table = Table([wave.to("micrometer"), spec1D, err, mask],
                  names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    table.write(output, format="fits", overwrite=True)
    return

def rebin_std(reftab, teltab, output, redo=False):
    """ Rebin the standard star spectra to match observations of science
    cubes. """
    if os.path.exists(output) and not redo:
        return
    refdata = Table.read(reftab)
    teldata = Table.read(teltab)
    wave = teldata["WAVE"].data
    flux = teldata["FLUX"].data
    flux[np.isnan(flux)] = 0.
    fluxerr = teldata["FLUX_ERR"].data
    refwave = refdata["WAVE"].data
    dw = np.diff(wave)[0]
    # Appending extra wavelength to cover the whole wavelength of the reference
    # spectrum
    if wave[0] >= refwave[0]:
        nbins = int((wave[0] - refwave[0]) / dw) + 5
        extrawave = wave[0] - np.arange(1, nbins)[::-1] * dw
        wave = np.hstack((extrawave, wave))
        flux = np.hstack((np.zeros(nbins), flux))
        fluxerr = np.hstack((np.zeros(nbins), fluxerr))
    if wave[0] <= refwave[0]:
        nbins = int((refwave[-1] - wave[-1]) / dw) + 5
        extrawave = wave[-1] + np.arange(1, nbins) * dw
        wave = np.hstack((wave, extrawave))
        flux = np.hstack((flux, np.zeros(nbins)))
        fluxerr = np.hstack((fluxerr, np.zeros(nbins)))
    newflux = spectres(refwave, wave, flux)
    newfluxerr = spectres(refwave, wave, fluxerr)
    mask = np.ones_like(refwave)
    mask[np.isnan(newflux)] = 0
    table = Table([refwave, newflux, newfluxerr, mask],
                  names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    table.write(output, overwrite=True)

if __name__ == "__main__":
    # Input files
    imgfile = os.path.join(context.home,
                           "data/HIP56736_combined_cubeImg_1.fits")
    stdcube = os.path.join(context.home, "data/HIP56736_combined_cube_1.fits")
    # Output table
    std_table = os.path.join(context.home, "data/HIP56736_spec1D.fits")
    # Extracting photometry
    std_phot(stdcube, imgfile, std_table, redo=True)
    # Rebin spectrum to match dispersion of data cube
    reftable = os.path.join(context.home, "data/spec1d_sn80/sn80_0001.fits")
    output = os.path.join(context.home, "data/molecfit/HIP56736_spec1D.fits")
    rebin_std(reftable, std_table, output, redo=True)

