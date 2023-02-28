# -*- coding: utf-8 -*-
"""

Created on 09/10/18

Author : Carlos Eduardo Barbosa
Updated by: R Elliot Meyer (2022/2023)

Run paintbox in the central part of M85 using WIFIS data.

"""

import os
import copy
import platform

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from matplotlib import cm
from spectres import spectres
import ppxf.ppxf_util as util
from tqdm import tqdm
import emcee
from scipy import stats
from scipy.interpolate import interp1d
from multiprocessing import Pool

import context
import paintbox as pb
import plotting

from sys import exit

import warnings
warnings.filterwarnings('ignore')

class CvDCaller():
    ''' 
    Class that defines the CvD models.

    Arguments
    ---------
    sed:        Paintbox ParametricModel or similar, built using CvD models.

    Methods
    ---------
    call:       Calls the input models with parameters theta 
    '''
    def __init__(self, sed):
        self.sed = sed
        self.parnames = list(dict.fromkeys(sed.parnames))
        self.wave = self.sed.wave
        self.nparams = len(self.parnames)
        self._shape = len(self.sed.parnames)
        self._idxs = {}
        for param in self.parnames:
            self._idxs[param] = np.where(np.array(self.sed.parnames) 
                                            == param)[0]

    def __call__(self, theta):
        '''Calls model with parameters (theta).
           Returns model with the input parameters
        '''
        t = np.zeros(self._shape)
        for param, val in zip(self.parnames, theta):
            t[self._idxs[param]] = val
        return self.sed(t)

def build_sed_CvD(wave, velscale=200, porder=45, elements=None, V=0,
                  templates_file=None, simple=False, short=False, alpha=False,
                  kinematic_fit=False, kinematic_priors = None):
    '''
    Builds a model SED using paintbox ParametricModel instances using CvD models
    and a various elemental abundances, kinematic constraints, and options.
    '''
                  
    if alpha:
        wdir = os.path.join(context.home, "templates_alpha")
        temp_file_old = os.path.join(context.home,
                                "templates_alpha/VCJ17_varydoublex_wifis.fits")
    else:
        wdir = os.path.join(context.home, "templates")
        temp_file_old = os.path.join(context.home,
                                "templates/VCJ17_varydoublex_wifis.fits")
    templates_file = temp_file_old if templates_file is None else templates_file

    elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K"] if \
                elements is None else elements
    if kinematic_fit or simple:
        elements = []

    # Loads template file, median normalizes templates
    #templates = fits.getdata(templates_file, ext=2)
    templates = fits.getdata(templates_file, ext=0)
    tnorm = np.median(templates, axis=1)
    templates /= tnorm[:, None]
    #params = Table.read(templates_file, hdu=3)
    params = Table.read(templates_file, hdu=1)
    if simple or kinematic_fit:
        idx = np.where(np.logical_and(params['x1'] == 1.3,
                    params['x2'] == 2.3))[0]
        params = params[idx]
        params = params[params.colnames[:2]]
        templates = templates[idx]
    else:
        params = params[params.colnames[:4]]

    # If fitting for age, select only models between 9-14 Gyr (?)
    # This code currently does not execute since the column name is "Age"
    if "age" in params.colnames:
        idx = np.where((params["age"] > 9) & (params["age"] < 14))[0]
        params = params[idx]
        templates = templates[idx]
        params.rename_column("age", "Age")
        params.rename_column("logzsol", "Z")

    # Load wavelength array data
    twave = Table.read(templates_file, hdu=2)["wave"].data
    priors = {}
    limits = {}

    # For each param, store max and minimum, then create uniform prior dist
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        if simple:
            if param == 'Age':
                vmax = 5.0
        limits[param] = (vmin, vmax)
        priors[param] = stats.uniform(loc=vmin, scale=vmax-vmin)
        #if simple:
        #    params.rename_column(param, param+'_Simple')
    ssp = pb.ParametricModel(twave, params, templates)

    # Load elemental information?
    for element in elements:
        elem_file = os.path.join(wdir, "C18_rfs_wifis_{}.fits".format(element))
        rfdata = fits.getdata(elem_file, ext=0)
        rfpar = Table.read(elem_file, hdu=1)
        #rfdata = fits.getdata(templates_file, extname="DATA.{}".format(element))
        #rfpar = Table.read(templates_file, hdu="PARS.{}".format(element))
        vmin, vmax = rfpar[element].min(), rfpar[element].max()
        limits[element] = (vmin, vmax)
        priors[element] = stats.uniform(loc=vmin, scale=vmax-vmin)
        #ewave = Table.read(templates_file, hdu=1)["wave"].data
        ewave = Table.read(elem_file, hdu=2)["wave"].data
        rf = pb.ParametricModel(ewave, rfpar, rfdata)
        ssp = ssp * pb.Resample(twave, rf)

    priors["Vsyst"] = stats.norm(loc=V, scale=100)
    #priors["Vsyst"] = stats.uniform(loc=V-100, scale=200)
    if kinematic_priors:
        sigma, width = kinematic_priors
        priors["sigma"] = stats.norm(loc=sigma, scale=width)
    else:
        priors["sigma"] = stats.uniform(loc=100, scale=400)

    if short:
        # Adding extinction to the stellar populations
        stars = pb.LOSVDConv(ssp)
        return CvDCaller(stars), priors
    else:
        # Adding extinction to the stellar populations
        stars = pb.Resample(wave, pb.LOSVDConv(ssp))

    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    for p in poly.parnames:
        if p == "p_0":
            mu, sd = 1, 0.3
            a, b = (0 - mu) / sd, (np.infty - mu) / sd
            priors[p] = stats.truncnorm(a, b, mu, sd)
        else:
            priors[p] = stats.norm(0, 0.02)

    # Using Polynomial to make sky model
    sky = pb.Polynomial(wave, 0)
    sky.parnames = [_.replace("p", "sky") for _ in sky.parnames]
    priors["sky_0"] = stats.norm(0, 0.1)

    # Creating a model including LOSVD
    sed = stars * poly + sky
    sed = CvDCaller(sed)
    missing = [p for p in sed.parnames if p not in priors.keys()]
    if len(missing) > 0:
        print("Missing parameters in priors: ", missing)
    else:
        print("No missing parameter in the model priors!")

    # theta = np.array([0, 10, 2., 2., 0, 0, 0, 0, 0.1, 3.8, 200, 729, 1])
    # theta = np.hstack([theta, np.zeros(porder + 1)])
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = params.colnames + elements
    sed.sspparams = params
    sed.porder = porder
    return sed, priors

def build_sed_model_emiles(wave, w1=8800, w2=13200, velscale=200, sample=None,
                           fwhm=2.5, porder=45, V=0):
    """ Build model for NGC 3311"""
    # Preparing templates
    sample = "all" if sample is None else sample
    templates_file = os.path.join(context.home, "templates",
        "emiles_wifis_vel{}_w{}_{}_{}_fwhm{}.fits".format(velscale, w1, w2,
                                        sample, fwhm))
    templates = fits.getdata(templates_file, ext=0)
    tnorm = np.median(templates, axis=1)
    templates /= tnorm[:, None]
    params = Table.read(templates_file, hdu=1)
    limits = {}
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        limits[param] = (vmin, vmax)
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave)
    ssp = pb.StPopInterp(twave, params, templates)
    # Adding extinction to the stellar populations
    extinction = pb.CCM89(twave)
    stars = pb.Rebin(wave, pb.LOSVDConv(ssp * extinction, velscale=velscale))
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    # Using Polynomial to make sky model
    sky = pb.Polynomial(wave, 20)
    sky.parnames = ["sky"]
    # Creating a model including LOSVD
    sed = stars * poly + sky
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = params.colnames
    sed.sspparams = params
    sed.porder = porder
    return sed

def run_sampler(loglike, priors, outdb, nsteps=3000):
    ndim = len(loglike.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(loglike.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(logpdf, theta)])
        if not np.isfinite(lp) or np.isnan(lp):
            return -np.inf
        return lp + loglike(theta)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True)
    return

def make_summary_table(trace, outtab):
    print("Saving results to summary table: ", outtab)
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def add_alpha(t, band="2mass_ks", quick=True):
    """ Uses the M/L table to the M/L and alpha parameters. 
        M/L table has a set of pre-calculated M/L values for 
        given metallicity/age/imf. The alpha parameter is 
        calculated for every sample point in the posterior 
        distribution to provide an estimate of alpha & M/L

        Inputs:
            t
    """
    outtab = copy.copy(t)
    krpa_imf1 = 1.3
    krpa_imf2 = 2.3
    krpa_imf3 = 2.3
    #ml_table = Table.read("/Users/meyer/WIFIS/paintbox/wifis/stpop/FSPS_magnitudes.fits")
    ml_table = Table.read(context.home + "FSPS_magnitudes.fits")
    ml_table = ml_table[ml_table["age"] > 0.98]

    if quick:
        ages = np.arange(1, 15)
        idxs = []
        for age in ages:
            diff = np.abs(ml_table["age"].data - age)
            idx = np.where(diff == diff.min())[0]
            idxs.append(idx)
        idxs = np.unique(np.hstack(idxs))
        ml_table = ml_table[idxs]

    m2ls = pb.ParametricModel(np.array([21635.6]),
                            ml_table["logzsol", "age", "imf1", "imf2"],
                            ml_table["ML_{}".format(band)].data)
    params = np.stack([t["Z"].data, t["Age"].data, t["x1"].data,
                    t["x2"].data]).T
    alphas = np.zeros(len(params))
    m2ltab = np.zeros(len(params))
    for i, p in enumerate(tqdm(params, desc="Calculating alpha parameter")):
        m2l = m2ls(p)
        m2l_kr = m2ls(np.array([p[0], p[1], krpa_imf1, krpa_imf2]))
        alphas[i] = m2l / m2l_kr
        m2ltab[i] = m2l
    outtab["M2L_Ks"] = m2ltab
    outtab["alpha_Ks"] = alphas
    return outtab

def run_paintbox(
        galaxy, 
        radius, 
        V, 
        date, 
        outdir, 
        velscale=200, 
        ssp_model="CvD",
        sample_emiles="all", 
        loglike="normal2", 
        elements=None,
        linefit = False, 
        nsteps=4000, 
        postprocessing=False, 
        porder=45, 
        testing=False, 
        kinematic_fit=False, 
        inflationary = False,
        kinematic_priors = None,
        ):

    # Defining fit parameters based on model
    if ssp_model == "CvD":
        corner_pars = ['Z', 'Age', 'x1', 'x2', 'Na', "Fe", 'Ca', "K"]
    elif ssp_model == "emiles":
        corner_pars = ['Z', 'T', 'x1', 'x2', 'Na', "Fe", 'Ca', "K"]
    if elements != None:
        adjusted_corner_pars = corner_pars[:3]
        for par in corner_pars[4:]:
            if par in elements:
                adjusted_corner_pars.append(par)
        corner_pars = adjusted_corner_pars
    
    #Grabbing cube filename
    cubename = "{}_combined_cube_1_telluricreduced_{}_{}.fits".format(
                galaxy, date, radius)
    #Define name for galaxy and region 
    name = "{} {}".format(galaxy, radius)
    # Determine fitting elements(?)
    elements_str = "all" if elements is None else "".join(elements)
    # Read first spectrum to set the dispersion
    flux = fits.getdata(cubename, ext=0)
    wave_lin = fits.getdata(cubename, ext=1)
    fluxerr = fits.getdata(cubename, ext=2)
    idx = np.where(np.isnan(flux), False, True)
    flux_interp = interp1d(wave_lin[idx], flux[idx], bounds_error=False,
                           fill_value=0)
    fluxerr_interp = interp1d(wave_lin[idx], fluxerr[idx], bounds_error=False,
                              fill_value=0)
    # Rebinning data
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]],
                                           flux, velscale=velscale)
    wave = np.exp(logwave)[20:-5]
    flux, fluxerr = spectres(wave, wave_lin, flux_interp(wave_lin),
                             spec_errs=fluxerr_interp(wave_lin))

    ###########################################################################
    # Uncertainty inflationary term for testing with Student's T loglike
    if inflationary and loglike == "studt":
        factor = 3. if radius=="R1" else 2.
        #factor = 1.
        fluxerr *= factor
    else:
        factor = 1.
    ###########################################################################

    # Normalizing the data
    norm = np.median(flux)
    flux /= norm
    fluxerr /= norm
    radius = cubename.split("_")[-1].split(".")[0]

    # Building the model
    print("Producing SED model with paintbox...")
    if ssp_model == "CvD":
        sed, priors = build_sed_CvD(wave, elements=elements, V=V,
                                    porder=porder,
                                    kinematic_fit=kinematic_fit,
                                    kinematic_priors=kinematic_priors)
    elif ssp_model == "emiles":
        sed = build_sed_model_emiles(wave, sample=sample_emiles)
    print("Done!")

    # Defining a mask for the fit
    regions = []
    if linefit:
        if galaxy == 'M85':
            lines = ['FeH','CaI','NaI','KI_a','KI_b','KI_1.25', 'PaB', 'NaI123']
        elif galaxy == 'NGC5557':
            lines = ['FeH','NaI','KI_a','KI_b','KI_1.25','PaB','NaI127']
        else:
            lines = context.line_name

        for i in range(len(context.bluelow)):
            if context.line_name[i] in lines:
                regions.append((context.bluelow[i],context.redhigh[i]))
        line_name = lines
    elif kinematic_fit:
        regions = [(8600,11550),(12200,13500)]
    else:
        #regions = [(9600,11000),(11400,13500)]#, (12900,13000)]
        #regions = [(8800,9600),(11000,11600)]#, (12900,13000)]
        #regions = [(8800,9600), (10040,10140),(10830,10845),(10900,11600),
        #            (12100,12160),(12650,12720),(12890,12940)]
        regions = []

    # Ones indicate masked region
    #sedmask = np.ones(len(sed.wave))
    sedmask = np.zeros(len(sed.wave))
    if len(regions) > 0:
        for region in regions:
            maskx = np.where((sed.wave >= region[0]) & 
                             (sed.wave <= region[1]))[0]
            sedmask[maskx] = 1

    if loglike == 'studt':
        logp = pb.StudTLogLike(flux, sed, obserr=fluxerr, mask=sedmask)
        priors["nu"] = stats.uniform(loc=2.01, scale=8)
    else:
        logp = pb.NormalLogLike(flux, sed, obserr=fluxerr, mask=sedmask)

    if testing:
        return sed, priors, logp, flux, fluxerr

    ############################################################################
    # Running Sampler
    outdb = os.path.join(outdir, "{}_{}_{}_{}_{}.h5".format(
                        galaxy, radius, ssp_model, loglike, elements_str))
    if not os.path.exists(outdb):
        print("Running emcee...")
        run_sampler(logp, priors, outdb, nsteps=nsteps)

    ############################################################################
    # Only do postprocessing analysis locally 
    if not postprocessing:
        return sed, priors, logp, flux, fluxerr
    ############################################################################

    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(.9 * nsteps), flat=True,
                                 thin=25)
    trace = Table(tracedata, names=logp.parnames)
    print("Using trace with {} samples".format(len(tracedata)))

    if kinematic_fit: # A bit of a temporary hack to remove issues in add_alpha
        trace.add_column(np.ones(len(trace))*1.3,name = 'x1')
        trace.add_column(np.ones(len(trace))*2.3,name = 'x2')
    trace = add_alpha(trace)
    if kinematic_fit: # Reversal of hack
        trace.remove_column('x1')
        trace.remove_column('x2')

    make_summary_table(trace, outtab)
    corner_name = outdb.replace(".h5", "_corner")
    plotting.plot_corner(trace, corner_pars, corner_name,
                title="{} {}".format(galaxy, radius), redo=True)
    print("Producing fitting figure...")
    trace = np.array(trace)
    plotting.plot_fitting(wave, flux, fluxerr / factor, sed, tracedata, trace, 
                          outdb, regions, context.line_name, redo=True, 
                          linefit = linefit, norm=norm, name=name, 
                          ylabel="Flux", reslabel="Res. (\%)", 
                          liketype=loglike)

if __name__ == "__main__":
    # CHANGE ALL SETTINGS IN THE context.py SCRIPT

    if platform.node() == 'wifis-monster':
        wdir = os.path.join(context.home, "wifis-data")
    else:
        wdir = os.path.join(context.home, "elliot")

    for galaxy in context.sample:
        galdir = os.path.join(wdir, galaxy)
        os.chdir(galdir)
        for radius in context.fit_regions:
            kin_priors = None
            if context.kin_priors:
                kin_priors = context.kin_priors[galaxy][radius]
            if context.forcedir:
                outdir = os.path.join(galdir, context.forcedir)
            else:
                outdir = os.path.join(galdir, radius+'_'+context.dirsuffix)
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            print("=" * 80)
            print("Processing galaxy {}, region {}".format(galaxy, radius))
            run_paintbox(galaxy, 
                         radius, 
                         context.V[galaxy], 
                         context.obsdates[galaxy][radius], 
                         outdir, 
                         ssp_model=context.ssp_model, 
                         loglike=context.loglike, 
                         elements=context.elements, 
                         postprocessing=context.postprocessing, 
                         nsteps=context.nsteps, 
                         porder=context.porder, 
                         linefit = False, 
                         kinematic_fit=True,
                         kinematic_priors = kin_priors)
