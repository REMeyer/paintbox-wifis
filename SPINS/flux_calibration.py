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
from tqdm import tqdm
import astropy.units as u
from astropy.table import Table
from spectres import spectres

import context

def apply_flux_calibration(obsstar, obsspec, outdir, redo=False,
                           reference=None, wmin=0.88, wmax=1.3, dmag=0.):
    if os.path.exists(outdir) and not redo:
        return
    reference = "Vega" if reference is None else reference
    observed = Table.read(obsstar)
    # Remove regions outside wavelength range
    idx = np.where((observed["WAVE"] > wmin) & (observed["WAVE"] < wmax))
    observed = observed[idx]
    # Reading template table
    if reference == "Vega":
        template = Table.read(os.path.join(context.data_dir,
                                           "rieke2008/table7.fits"))
        template.rename_column("lambda", "WAVE")
        template.rename_column("Vega", "FLUX")
    else:
        pass
    # Cropping template in wavelength
    idx = np.where((template["WAVE"] > wmin) & (template["WAVE"] < wmax))
    template = template[idx]
    ############################################################################
    # Scaling the flux of Vega to that of the standard star
    stdflux = template["FLUX"] * np.power(10, -0.4 * dmag)
    # Determining the sensitivity function
    sensfun = calc_sensitivity_function(observed["WAVE"], observed["tacflux"],
                            template["WAVE"], stdflux)
    # Applying sensitivity function to galaxy spectra
    print("Applying flux calibration...")
    for spec in tqdm(obsspec):
        table = Table.read(spec)
        idx = np.where((table["WAVE"] > wmin) & (table["WAVE"] < wmax))
        table = table[idx]
        wave = table["WAVE"] * u.micrometer
        newflux = table["tacflux"].data * sensfun(wave) * template["FLUX"].unit
        newfluxerr = table["tacdflux"].data * sensfun(wave) * template[
            "FLUX"].unit
        newtable = Table([wave, newflux, newfluxerr], names=["WAVE", "FLUX",
                                                             "FLUX_ERR"])
        newtable.write(os.path.join(outdir, os.path.split(spec)[1]),
                       overwrite=True)


def calc_sensitivity_function(owave, oflux, twave, tflux, order=30):
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
