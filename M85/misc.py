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