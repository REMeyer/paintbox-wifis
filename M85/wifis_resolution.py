# -*- coding: utf-8 -*-
""" 

Created on 18/10/18

Author : Carlos Eduardo Barbosa

Determination of WIFIS properties (based on their paper) to determine
reasonable sampling properties for the analysis of the kinematics.

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import constants

def plot_wifis_vel_resolution(mode=None):
    mode = "zJ" if mode is None else mode
    Rs = {"zJ": 2500, "Hs": 3000}
    wrange = {"zJ":(0.9 * u.micrometer, 1.35 * u.micrometer),
              "Hs": (1.5 * u.micrometer, 1.7 * u.micrometer)}
    wave = np.linspace(wrange[mode][0], wrange[mode][1], 1000)
    R = Rs[mode]
    fwhm = wave / R
    velscale = constants.c.to("km/s") * fwhm / wave / 2.634
    print(velscale)
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, velscale, "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Velocity scale - sigma (km/s)")
    plt.show()

if __name__ == "__main__":
    plot_wifis_vel_resolution()