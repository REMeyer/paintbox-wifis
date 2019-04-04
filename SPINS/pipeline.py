# -*- coding: utf-8 -*-
""" 

Created on 04/04/19

Author : Carlos Eduardo Barbosa

Pipeline for the analysis of SPINS galaxies

1) Uses a configuration file to determine the parameters of the analysis
2) Produces Voronoi binning of the galaxies
3) Extracts the spectrum of a telluric standard star for calibration
4) Make telluric correction using molecfit
5) Makes flux calibration

"""
from __future__ import print_function, division

import os
import yaml

from astropy.io import fits

import context
from make_voronoi import make_voronoi, combine_spectra
from stdphot import stdphot, rebinstd

if __name__ == "__main__":
    home_dir = os.path.join(context.data_dir, "WIFIS")
    config_file = "sn30_mom2.yaml"
    for gal in os.listdir(home_dir):
        data_dir = os.path.join(home_dir, gal)
        os.chdir(data_dir)
        if config_file not in os.listdir("."):
            continue
        with open(config_file) as f:
            params = yaml.load(f)
        # Make output table for a given configuration file
        outdir = os.path.join(home_dir, gal, config_file.replace(".yaml", ""))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        ########################################################################
        # Producing Voronoi binning
        datacube = os.path.join(data_dir, params["datacube"])
        dataheader = os.path.join(data_dir, params["datacube"])
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        vorfile = os.path.join(outdir, "voronoi.fits")
        make_voronoi(datacube, params["vorSN"], vorfile, redo=False)
        specs_dir = os.path.join(outdir, "combined")
        if not os.path.exists(specs_dir):
            os.mkdir(specs_dir)
        combine_spectra(datacube, vorfile, specs_dir, redo=False)
        ########################################################################
        # Extracting spectrum for telluric correction and flux calibration
        stdcube = os.path.join(data_dir, params["stdcube"])
        stdimg = os.path.join(data_dir, params["stdimg"])
        stdoutdir = os.path.join(outdir, "molecfit")
        if not os.path.exists(stdoutdir):
            os.mkdir(stdoutdir)
        stdout = os.path.join(stdoutdir, "{}.fits".format(
                              params["stdcube"].split("_")[0]))
        stdphot(stdcube, stdimg, stdout, r=params["tell_extract_radius"],
                redo=True)
        # Rebin standard star spectrum to match that of the data cube
        specs = sorted(os.listdir(specs_dir))
        reftable = os.path.join(specs_dir, specs[0])
        stdrebin = stdout.replace(".fits", "_rebin.fits")
        rebinstd(reftable, stdout, stdrebin, redo=True)
        ########################################################################


