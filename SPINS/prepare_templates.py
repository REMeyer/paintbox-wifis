# -*- coding: utf-8 -*-
""" 

Created on 05/04/19

Author : Carlos Eduardo Barbosa

Routine to prepare templates for pPXF fitting

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack
import ppxf.ppxf_util as util
from tqdm import tqdm

import misc

class EMILES():
    def __init__(self):
        """ Class to handle EMILES models. """
        self.data_dir = "/home/kadu/Dropbox/SPLUS/ifusci/EMILES"
        self.bis = np.array([0.30, 0.50, 0.80, 1.00, 1.30,
                                  1.50, 1.80, 2.00, 2.30, 2.50,
                                  2.80, 3.00, 3.30, 3.50])
        self.Zs = np.array([-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35,
                            -0.25,  0.06,  0.15,  0.26,  0.4 ])
        self.Ts = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15,
                            0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8,
                            0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                            3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
                            7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
                            11.5, 12.0, 12.5, 13.0, 13.5, 14.0])
        return

    def filename(self, age, metal, imf=None):
        """ Retrieve the filename for a given age and metallicity. """
        imf = 1.3 if imf is None else imf
        msign = "p" if metal >= 0. else "m"
        azero = "0" if age < 10. else ""
        fname = "Ebi{0:.2f}Z{1}{2:.2f}T{3}{4:02.4f}_iTp0.00_baseFe.fits".format(
            imf, msign, abs(metal), azero, age)
        return os.path.join(self.data_dir, fname)

def prepare_templates(params, outfile, redo=False):
    """ Prepares the templates for the fitting using EMILES SSPs."""
    if os.path.exists(outfile) and not redo:
        return
    emiles = EMILES()
    wmin = params["wmin"] * u.micrometer
    wmax = params["wmax"] * u.micrometer
    # Modify wmin to compensate for the recession velocity of the system
    zmax = (params["vsyst"] + 3000) / const.c.to("km/s").value
    wrest = wmin / (1 + zmax)
    grid = np.array(np.meshgrid(params["ages"], params["metals"],
                                params["bis"])).T.reshape(-1, 3)
    ssppars = Table(grid, names=["T", "Z", "imf"])
    filenames = []
    for args in grid:
        filenames.append(os.path.join(emiles.data_dir,
                                      emiles.filename(*args)))
    wave, spec = misc.read_spec(filenames[0])
    wave = wave * u.angstrom
    idx = np.where((wave > wrest) & (wave <= wmax))
    wave = wave[idx]
    spec = spec[idx]
    wrange = [wave[0].to("angstrom").value, wave[-1].to("angstrom").value]
    newflux, logLam, velscale = util.log_rebin(wrange, spec,
                                               velscale=params["velscale"])
    ssps = np.zeros((len(filenames), len(newflux)))
    norms = np.zeros(len(filenames))
    print("Processing SSP files")
    for i, fname in tqdm(enumerate(filenames)):
        spec = fits.getdata(fname)[idx]
        newflux, logLam, velscale = util.log_rebin(wrange, spec,
                                                   velscale=params["velscale"])
        norm = np.median(newflux)
        ssps[i] = newflux / norm
        norms[i] = norm
    hdu1 = fits.PrimaryHDU(ssps)
    hdu1.header["EXTNAME"] = "SSPS"
    norms = Table([norms], names=["norm"])
    ssppars = hstack((ssppars, norms))
    hdu2 = fits.BinTableHDU(ssppars)
    hdu2.header["EXTNAME"] = "PARAMS"
    hdu1.header["CRVAL1"] = logLam[0]
    hdu1.header["CD1_1"] = logLam[1] - logLam[0]
    hdu1.header["CRPIX1"] = 1.
    # Making wavelength array
    hdu3 = fits.BinTableHDU(Table([logLam], names=["loglam"]))
    hdu3.header["EXTNAME"] = "LOGLAM"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(outfile, overwrite=True)
    return

