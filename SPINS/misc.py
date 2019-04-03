# -*- coding: utf-8 -*-
""" 

Created on 19/10/18

Author : Carlos Eduardo Barbosa

Miscelaneous functions.

"""
import numpy as np

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
