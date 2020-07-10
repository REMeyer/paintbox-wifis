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

def run_ppxf(specs, templates_file, config, outdir, redo=False, mask=None,
             plot=False):
    """ Run pPXF in all spectra. """
    velscale = config["velscale"]
    ssp_templates = fits.getdata(templates_file, extname="SSPS").T
    params = Table.read(templates_file, hdu=1)
    # if regul:
    #     ssp_templates, reg_dim, params = make_regul_array(ssp_templates, params)
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
        print(imgfile)
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
        ########################################################################
        # Applying mask
        lam = np.exp(logLam)
        goodpixels = np.arange(len(lam))
        if mask is not None:
            for w in mask:
                goodpix = np.argwhere((lam <= w[0]) | (lam >= w[1])).ravel()
                goodpixels = np.intersect1d(goodpixels, goodpix)
        pp0 = ppxf(templates, galaxy, noise, velscale=velscale,
                  plot=False, moments=config["nmoments"], start=start0,
                  vsyst=dv, lam=lam, component=components,
                  degree=config["degree"], quiet=False, sky=sky,
                  mdegree=config["mdegree"], clean=False, goodpixels=goodpixels)
        plt.clf()
        X = pp0.galaxy - pp0.bestfit
        medX = np.nanmedian(X)
        mad = np.median(np.abs(X - medX))
        std = 1.4826 * mad
        noise = np.ones_like(pp0.galaxy) * std
        pp = ppxf(templates, galaxy, noise, velscale=velscale,
                  plot=True, moments=config["nmoments"], start=start0,
                  vsyst=dv, lam=np.exp(logLam), component=components,
                  degree=config["degree"], quiet=False, sky=sky,
                  mdegree=config["mdegree"], clean=config["clean"],
                  goodpixels = goodpixels)
        plt.show()
        X = pp.galaxy - pp.bestfit
        medX = np.nanmedian(X)
        mad = np.median(np.abs(X - medX))
        std = 1.4826 * mad
        noise = np.ones_like(pp0.galaxy) * std
        signal = np.median(pp0.galaxy)
        sn = signal / noise
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
        pp.sn = float(sn.mean())
        pp.wsky = float(pp.weights[-1])
        # Saving the weights of the bestfit
        idx = Table([np.arange(len(weights))], names=["idx"])
        wtable = hstack([idx, params[params.colnames[:-1]], weights])
        i = np.argwhere(wtable["mass_weight"] > 0).ravel()
        wtable = wtable[i]
        wtable.write(wfile, overwrite=True)
        #
        array_keys = ["lam", "galaxy", "noise", "bestfit", "mpoly", "apoly"]
        table = Table([getattr(pp, key) for key in array_keys],
                      names=array_keys)
        table.write(afile, overwrite=True)
        # Saving results and plot
        save(pp, ppfile)
        plt.savefig(imgfile, dpi=400)
        plt.clf()

def save(pp, output):
    """ Save results from pPXF into files excluding fitting arrays. """
    ppdict = {}
    save_keys = ["regul", "degree", "mdegree", "reddening", "clean", "ncomp",
                 "chi2", "nonzero_ssps", "nssps", "flux_norm", "sn",
                 "wsky"]
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

def make_ppxf_table(vorfile, ppdir, keys, output, redo=False):
    """ Produces a table with the ppxf results. """
    if os.path.exists(output) and not redo:
        return
    yamls = sorted([_ for _ in os.listdir(ppdir) if _.endswith("yaml")])
    data = []
    for yfile in yamls:
        with open(os.path.join(ppdir, yfile)) as f:
            info = yaml.load(f)
        d = []
        for key in keys:
            if key in info.keys():
                d.append(info[key])
            else:
                d.append(np.nan)
        data.append(d)
    data = np.array(data)
    newdata = Table(data, names=keys)
    vordata = fits.getdata(vorfile, ext=0)
    tabdata = Table.read(vorfile)
    newtable = hstack([tabdata, newdata])
    hvor = fits.getheader(vorfile)
    vorHDU = fits.PrimaryHDU(vordata, header=hvor)
    tabHDU = fits.BinTableHDU(newtable)
    hdulist = fits.HDUList([vorHDU, tabHDU])
    hdulist.writeto(output, overwrite=True)
    return