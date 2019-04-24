# -*- coding: utf-8 -*-
""" 

Created on 19/10/18

Author : Carlos Eduardo Barbosa

Miscelaneous functions.

"""
import numpy as np
from astropy.io import fits

def calc_wave(header, naxis=1):
    wave = ((np.arange(header['NAXIS3']) + 1
             - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3'])


def snr(flux, axis=0):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro """
    signal = np.nanmedian(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.*flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)
    return signal, noise, signal / noise

def read_spec(filename, axis=1, extension=0):
    """ Produces array for wavelenght of a given array. """
    h = fits.getheader(filename, ext=extension)
    w0 = fits.getval(filename, "CRVAL{0}".format(axis), extension)
    if "CD{0}_{0}".format(axis) in list(h.keys()):
        deltaw = fits.getval(filename, "CD{0}_{0}".format(axis), extension)
    else:
        deltaw = fits.getval(filename, "CDELT{0}".format(axis), extension)
    pix0 = fits.getval(filename, "CRPIX{0}".format(axis), extension)
    npix = fits.getval(filename, "NAXIS{0}".format(axis), extension)
    wave = w0 + deltaw * (np.arange(npix) + 1 - pix0)
    flux = fits.getdata(filename, extension)
    return wave, flux
