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

def extract_telluric_std_spectrum(plot=False):
    """Determination of aperture and extraction of 1D spectrum. """
    ############################################################################
    # Defining the aperture with maximum S/N using the image as a reference
    imgfile = os.path.join(context.home,
                           "data/HIP56736_combined_cubeImg_1.fits")
    data = fits.getdata(imgfile)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    daofind = DAOStarFinder(fwhm=2.1, threshold=10.*std)
    star= daofind(data - median)
    positions = [(star["xcentroid"][0], star["ycentroid"][0])]
    radii = np.linspace(0.5,10, 100)
    apertures = [CircularAperture(positions, r=r) for r in radii]
    phot_table = aperture_photometry(data-median, apertures)
    flux = np.array([phot_table["aperture_sum_{}".format(i)].data[0] for i
                       in range(len(radii))])
    sn = flux / np.sqrt((flux-median) + (std * np.pi * radii**2)**2)
    idx = np.where(sn == sn.max())
    # See http://wise2.ipac.caltech.edu/staff/fmasci/GaussApRadius.pdf
    best_rad = radii[idx] / 0.673
    ############################################################################
    # Extracting the spectra from the datacube
    datacube = os.path.join(context.home, "data/HIP56736_combined_cube_1.fits")
    cube = fits.getdata(datacube)
    zdim, ydim, xdim = cube.shape
    x = np.arange(xdim) + 1
    y = np.arange(ydim) + 1
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx - star["xcentroid"])**2 + (yy - star["ycentroid"])**2)
    idx = np.where(r <= best_rad)
    specs = cube[:, idx[0], idx[1]]
    hdr = fits.getheader(datacube)
    wave = ((np.arange(hdr['NAXIS3']) + 1 - hdr['CRPIX3']) * hdr['CDELT3'] + \
           hdr['CRVAL3']) * u.m
    idxnorm = np.where((wave.to(u.AA).value > 10300) &
                       (wave.to(u.AA).value < 10600))
    norm = np.nanmean(specs[idxnorm,:], axis=1)
    specs = specs / norm[:, None]
    spec1D = np.nanmedian(specs[0], axis=1)
    err = np.zeros_like(spec1D)
    mask = np.ones_like(spec1D)
    mask[np.isnan(spec1D)] = 0.
    table = Table([wave.to("micrometer"), spec1D, err, mask],
                  names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    output = os.path.join(context.home,
                           "data/HIP56736_spec1D.fits")
    table.write(output, format="fits", overwrite=True)
    # Plot used to check the results
    if plot:
        rad = r[idx]
        for i, spec in enumerate(specs.T):
            plt.plot(wave.to("micrometer"), spec, "-",
                     label="r={:.2f}".format(rad[i]))
        plt.plot(wave.to("micrometer"), spec1D, "-k", label="median")
        plt.legend(ncol=3)
        plt.show()

if __name__ == "__main__":
    extract_telluric_std_spectrum(plot=True)

