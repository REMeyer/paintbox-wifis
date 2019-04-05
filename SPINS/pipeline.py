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
from datetime import datetime

import numpy as np
from astropy.io import fits
from astroquery.vizier import Vizier

import context
from make_voronoi import make_voronoi, combine_spectra
from stdphot import stdphot, rebinstd
from run_molecfit import run_molecfit
from flux_calibration import apply_flux_calibration

if __name__ == "__main__":
    home_dir = os.path.join(context.data_dir, "WIFIS")
    config_file = "sn30_mom2.yaml"
    # Setting up query of
    two_mass = Vizier(columns=["*", "+_r"])
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
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        stdout = os.path.join(outdir, "{}.fits".format(
                              params["stdcube"].split("_")[0]))
        stdphot(stdcube, stdimg, stdout, r=params["tell_extract_radius"],
                redo=True)
        # Rebin standard star spectrum to match that of the data cube
        specs = sorted(os.listdir(specs_dir))
        specs = [os.path.join(specs_dir, _) for _ in specs]
        filelist = os.path.join(outdir, "filelist.txt")
        with open(filelist, "w") as f:
            f.write("\n".join(specs))
        reftable = os.path.join(specs_dir, specs[0])
        stdrebin = stdout.replace(".fits", "_rebin.fits")
        rebinstd(reftable, stdout, stdrebin, redo=False)
        ########################################################################
        # Uses molecfit to make telluric correction
        stdheader = fits.getheader(stdimg)
        jd =  float(stdheader["BARY_JD"])
        mjd = int(jd - 2400000.5)
        utc = datetime.strptime("12:19:32.5", "%H:%M:%S.%f")
        utc_seconds = utc.hour * 3600 + utc.minute * 60 + utc.second
        molecfit_params = {"user_workdir": outdir, "filename" : stdrebin,
                           "listname" : "filelist.txt",
                           "obsdate" : mjd, "utc" : utc_seconds,
                           "telalt" : stdheader["ELEVATIO"]}
        molecfit_dir = os.path.join(molecfit_params["user_workdir"], "molecfit")
        run_molecfit(molecfit_params, molecfit_dir, redo=False)
        ########################################################################
        # Make flux calibration
        fcalib_dir = os.path.join(outdir, "fcalib")
        if not os.path.exists(fcalib_dir):
            os.mkdir(fcalib_dir)
        stdfile = os.path.join(molecfit_dir,
                  os.path.split(stdrebin.replace(".fits", "_TAC.fits"))[1])
        specs = [os.path.join(molecfit_dir, _.replace(".fits", "_TAC.fits"))
                 for _ in sorted(os.listdir(specs_dir))]
        stdname = fits.getval(params["stdcube"], "OBJECT")
        std2mass = two_mass.query_object(stdname, catalog="II/246")[0][0]
        ref2mass = two_mass.query_object(params["fcalib_template"],
                                         catalog="II/246")[0][0]
        dmag = std2mass["Kmag"] - ref2mass["Kmag"]
        dmagerr = np.sqrt(std2mass["e_Kmag"]**2 + ref2mass["e_Kmag"]**2)
        apply_flux_calibration(stdfile, specs, fcalib_dir, redo=False,
                               wmin=params["wmin"], wmax=params["wmax"],
                               reference=params["fcalib_template"],
                               dmag=dmag)
        ########################################################################



