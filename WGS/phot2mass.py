# -*- coding: utf-8 -*-
""" 

Created on 23/04/19

Author : Carlos Eduardo Barbosa

Uses MGE to model the photometry of 2MASS images of SPINS galaxies

"""
from __future__ import print_function, division

import os

from astropy.io import fits
import matplotlib.pyplot as plt
from mgefit.find_galaxy import find_galaxy
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours

import context

if __name__ == "__main__":
    data_dir = os.path.join(context.data_dir, "2MASS")
    for galaxy in os.listdir(data_dir):
        jband_file = os.path.join(data_dir, galaxy, "m85_mosaic_j.fits")
        ngauss = 20
        sigmapsf = 2
        normpsf = 2
        scale=1
        jband = fits.getdata(jband_file)
        geom = find_galaxy(jband, fraction=0.05)
        s = sectors_photometry(jband, geom.eps, geom.theta, geom.xpeak,
                               geom.ypeak, minlevel=1., plot=False)
        m = mge_fit_sectors(s.radius, s.angle, s.counts, geom.eps,
                            ngauss=ngauss, sigmapsf=sigmapsf, normpsf=normpsf,
                            scale=scale, plot=1, bulge_disk=0, linear=0)

        plt.show()