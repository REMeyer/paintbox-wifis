# -*- coding: utf-8 -*-
""" 

Created on 19/02/19

Author : Carlos Eduardo Barbosa

Program to determine sensitivity function of observations and application to
observed spectra.

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy import constants
from astropy.table import Table
import matplotlib.pyplot as plt
from spectres import spectres

import context

def calc_sensitivity_function(owave, oflux, twave, tflux, order=40):
    """ Calculates the sensitivity function using a polynomial approximation.
    """

    # Setting the appropriate wavelength regime
    wmin = np.maximum(owave[1], twave[1])
    wmax = np.minimum(owave[-2],twave[-2])
    dw = 0.1 * np.minimum(owave[1] - owave[0], twave[1] - twave[0])
    wave = np.arange(wmin, wmax, dw)
    # Rebinning and normalizing spectra
    oflux = spectres(wave, owave, oflux)
    tflux = spectres(wave, twave, tflux)
    sens = np.poly1d(np.polyfit(wave, tflux / oflux, order))
    return sens


if __name__ == "__main__":
    observed = Table.read(os.path.join(context.home,
                          "data/molecfit/output/HIP56736_spec1D_TAC.fits"))
    wmin = 0.885
    wmax = 1.300
    # Remove zeros from spectrum
    idx = np.where((observed["WAVE"] > wmin) & (observed["WAVE"] < wmax))
    observed = observed[idx]
    # Reading template table
    template = Table.read(os.path.join(context.home, "rieke2008/table7.fits"))
    # Cropping template in wavelength
    idx = np.where((template["lambda"] > wmin) & (template["lambda"] < wmax))
    template = template[idx]
    # Scaling the flux of Vega to that of the standard star
    deltamag = 8.857
    stdflux = template["Vega"] * np.power(10, -0.2 * deltamag)
    # Determining the sensitivity function
    sensfun = calc_sensitivity_function(observed["WAVE"], observed["tacflux"],
                            template["lambda"], stdflux)
    # Applying sensitivity function to galaxy spectra
    targetSN = 40
    wdir = os.path.join(context.home, "data/molecfit_sn{}".format(targetSN))
    outdir = os.path.join(context.home, "data/fcalib_sn{}".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for filename in sorted(os.listdir(wdir)):
        table = Table.read(os.path.join(wdir, filename))
        idx = np.where((table["WAVE"] > wmin) & (table["WAVE"] < wmax))
        table = table[idx]
        wave = table["WAVE"] * u.micrometer
        newflux = table["tacflux"].data * sensfun(wave) * template["Vega"].unit
        newfluxerr = table["tacdflux"].data * sensfun(wave) * template[
            "Vega"].unit
        newtable = Table([wave, newflux, newfluxerr], names=["WAVE", "FLUX",
                                                             "FLUX_ERR"])
        newtable.write(os.path.join(outdir, filename), overwrite=True)