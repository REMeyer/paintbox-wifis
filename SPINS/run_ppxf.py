# -*- coding: utf-8 -*-
"""
@author: Carlos Eduardo Barbosa

Run pPXF in data
"""
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy import constants
from astropy.table import Table, hstack
from ppxf import ppxf_util
from ppxf.ppxf import ppxf

import misc

def run_ppxf(specs, templates_file, config, outdir, redo=False, regul=False):
    """ Run pPXF in all spectra. """
    velscale = config["velscale"]
    ssp_templates = fits.getdata(templates_file, extname="SSPS").T
    params = Table.read(templates_file, hdu=1)
    if regul:
        ssp_templates, reg_dim, params = make_regul_array(ssp_templates, params)
    nssps = ssp_templates.shape[1]
    logwave_temp = Table.read(templates_file, hdu=2)
    start0 = [config["vsyst"], 100.]
    if config["nmoments"] > 2:
        nextra = config["nmoments"] - 2
        for n in range(nextra):
            start0.append(0.)
    w1 = config["wmin"] * u.micrometer
    w2 = config["wmax"] * u.micrometer
    for spec in specs:
        fname = os.path.split(spec)[-1]
        wfile = os.path.join(outdir, fname.replace(".fits", "_weights.fits"))
        afile = os.path.join(outdir, fname.replace(".fits", "_bestfit.fits"))
        ppfile = os.path.join(outdir, fname.replace(".fits", ".yaml"))
        imgfile = os.path.join(outdir, fname.replace(".fits", ".png"))
        if os.path.exists(ppfile) and not redo:
            continue
        # Reading the data in the files
        tab = Table.read(spec)
        wave = tab["WAVE"].data * u.micrometer
        flux = tab["FLUX"]
        fluxerr = tab["FLUX_ERR"]
        skyfile = spec.replace("spec", "sky")
        if os.path.exists(skyfile):
            sky = Table.read(skyfile)["FLUX"].data
        else:
            sky = None
        #######################################################################
        # Trim spectra to conform to the wavelenght range of the templates
        idx = np.where((wave > w1) & (wave < w2))
        wave = wave[idx]
        flux = flux[idx]
        fluxerr = fluxerr[idx]
        ########################################################################
        # Normalize spectrum to set regularization to a reasonable scale
        flux_norm = float(np.median(flux))
        flux /= flux_norm
        fluxerr /= flux_norm
        ########################################################################
        if np.all(fluxerr==0.):
            print("Error in flux is zero, estimating S/N from input spectrum.")
            signal, noise, sn = misc.snr(flux)
            fluxerr = np.ones_like(fluxerr) * noise
        # Rebinning the data to a logarithmic scale for ppxf
        wave_range = [wave[0].to("angstrom").value,
                      wave[-1].to("angstrom").value]
        galaxy, logLam, vtemp = ppxf_util.log_rebin(wave_range,
                                               flux.data, velscale=velscale)
        noise = ppxf_util.log_rebin(wave_range, fluxerr,
                               velscale=velscale)[0]
        ########################################################################
        # Preparing sky data if it exists
        if sky is not None:
            sky = sky[idx]
            sky /= flux_norm
            sky = ppxf_util.log_rebin(wave_range, sky, velscale=velscale)[0]
        ########################################################################
        # Setting up the gas templates
        # gas_templates, line_names, line_wave = \
        #     ppxf_util.emission_lines(logwave_temp["loglam"].data,
        #                              [wave[0].value, wave[-1].value],
        #                              context.FWHM)
        # ngas = gas_templates.shape[1]
        ########################################################################
        # Preparing the fit
        dv = (logwave_temp["loglam"][0] - logLam[0]) * \
             constants.c.to("km/s").value
        components = np.zeros(nssps).astype(int)
        templates = ssp_templates
        # templates = np.column_stack((ssp_templates, gas_templates))
        # components = np.hstack((np.zeros(nssps), np.ones(ngas))).astype(np.int)
        # gas_component = components > 0
        ########################################################################
        # Fitting with two components
        pp = ppxf(templates, galaxy, noise, velscale=velscale,
                  plot=True, moments=config["nmoments"], start=start0,
                  vsyst=dv, lam=np.exp(logLam), component=components,
                  degree=config["degree"], quiet=False, sky=sky,
                  mdegree=config["mdegree"], clean=config["clean"])
        # Calculating average stellar populations
        weights = Table([pp.weights[:nssps] * params["norm"]], names=[
            "mass_weight"])
        for colname in params.colnames[:-1]:
            mean = np.average(params[colname], weights=weights[
                "mass_weight"].data)
            setattr(pp, colname, float(mean))
        # Including additional info in the pp object
        pp.nssps = nssps
        pp.nonzero_ssps = np.count_nonzero(weights)
        pp.flux_norm = flux_norm
        pp.colnames = params.colnames[:-1]
        # Saving the weights of the bestfit
        wtable = hstack([params[params.colnames[:-1]], weights])
        wtable.write(wfile, overwrite=True)
        #
        array_keys = ["lam", "galaxy", "noise", "bestfit", "mpoly", "apoly"]
        table = Table([getattr(pp, key) for key in array_keys],
                      names=array_keys)
        table.write(afile, overwrite=True)
        # Saving results and plot
        save(pp, ppfile)
        plt.savefig(imgfile, dpi=250)
        plt.clf()

def save(pp, output):
    """ Save results from pPXF into files excluding fitting arrays. """
    ppdict = {}
    save_keys = ["regul", "degree", "mdegree", "reddening", "clean", "ncomp",
                 "chi2", "nonzero_ssps", "nssps", "flux_norm"]
    save_keys += pp.colnames
    # Chi2 is a astropy.unit.quantity object, we have to make it a scalar
    pp.chi2 = float(pp.chi2)
    for key in save_keys:
        ppdict[key] = getattr(pp, key)
    klist = ["V", "sigma", "h3", "h4", "h5", "h6"]
    pp.sol = np.atleast_2d(pp.sol)
    pp.error = np.atleast_2d(pp.error)
    for j, sol in enumerate(pp.sol):
        for i in range(len(sol)):
            ppdict["{}_{}".format(klist[i], j)] = float(sol[i])
            ppdict["{}err_{}".format(klist[i], j)] = float(pp.error[j][i])

    with open(output, "w") as f:
        yaml.dump(ppdict, f, default_flow_style=False)