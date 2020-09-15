# -*- coding: utf-8 -*-
"""

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Run paintbox in the central part of M85 using WIFIS data.

"""

import os
import copy

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from matplotlib import cm
from spectres import spectres
try:
    import ppxf_util as util
except:
    import ppxf.ppxf_util as util
import pymc3 as pm
import theano.tensor as tt
from tqdm import tqdm
import seaborn as sns
import emcee
from scipy import stats
from scipy.interpolate import interp1d

import context
import paintbox as pb

class CvDCaller():
    def __init__(self, sed):
        self.sed = sed
        self.parnames = list(dict.fromkeys(sed.parnames))
        self.wave = self.sed.wave
        self.nparams = len(self.parnames)
        self._shape = len(self.sed.parnames)
        self._idxs = {}
        for param in self.parnames:
            self._idxs[param] = np.where(np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        t = np.zeros(self._shape)
        for param, val in zip(self.parnames, theta):
            t[self._idxs[param]] = val
        return self.sed(t)

def build_sed_CvD(wave, velscale=200, porder=45, elements=None):
    wdir = os.path.join(context.home, "templates")
    elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K"] if \
                elements is None else elements
    ssp_file = os.path.join(context.home,
                            "templates/VCJ17_varydoublex_wifis.fits")
    templates = fits.getdata(ssp_file, ext=0)
    tnorm = np.median(templates, axis=1)
    templates /= tnorm[:, None]
    params = Table.read(ssp_file, hdu=1)
    twave = Table.read(ssp_file, hdu=2)["wave"].data
    limits = {}
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        limits[param] = (vmin, vmax)
    ssp = pb.StPopInterp(twave, params, templates)
    ssppars = params.colnames + ["Na"]
    for element in elements:
        print(element)
        elem_file = os.path.join(wdir, "C18_rfs_wifis_{}.fits".format(element))
        rfdata = fits.getdata(elem_file, ext=0)
        rfpar = Table.read(elem_file, hdu=1)
        vmin, vmax = rfpar[element].min(), rfpar[element].max()
        limits[element] = (vmin, vmax)
        ewave = Table.read(elem_file, hdu=2)["wave"].data
        rf = pb.StPopInterp(ewave, rfpar, rfdata)
        ssp = ssp * rf
    # Adding extinction to the stellar populations
    stars = pb.Rebin(wave, pb.LOSVDConv(ssp, velscale=velscale))
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    # Using Polynomial to make sky model
    sky = pb.Polynomial(wave, 0)
    sky.parnames = [_.replace("p", "sky") for _ in sky.parnames]
    # Creating a model including LOSVD
    sed = stars * poly + sky
    sed = CvDCaller(sed)
    # theta = np.array([0, 10, 2., 2., 0, 0, 0, 0, 0.1, 3.8, 200, 729, 1])
    # theta = np.hstack([theta, np.zeros(porder + 1)])
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = ssppars
    sed.sspparams = params
    sed.porder = porder
    return sed

def build_sed_model_emiles(wave, w1=8800, w2=13200, velscale=200, sample=None,
                           fwhm=2.5, porder=30):
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

def make_pymc3_model(flux, sed, loglike=None, fluxerr=None):
    loglike = "normal2" if loglike is None else loglike
    flux = flux.astype(np.float)
    fluxerr = np.ones_like(flux) if fluxerr is None else fluxerr
    model = pm.Model()
    polynames = ["p{}".format(i + 1) for i in range(sed.porder)]
    params = np.unique(sed.parnames)
    with model:
        vars = {}
        for param in np.unique(sed.parnames):
            # Stellar population parameters
            if param in sed.ssppars:
                vmin, vmax = sed.ssppars[param]
                vinit = float(0.5 * (vmin + vmax))
                v = pm.Uniform(param, lower=float(vmin), upper=float(vmax),
                               testval=vinit)
                vars[param] = v
            # Dust attenuation parameters
            elif param == "Av":
                v = pm.Exponential("Av", lam=1 / 0.4, testval=0.1)
            elif param == "Rv":
                BNormal = pm.Bound(pm.Normal, lower=0)
                v = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
            elif param == "V":
                # Stellar kinematics
                v= pm.Normal("V", mu=729., sd=50., testval=729)
            elif param == "sigma":
                v = pm.Uniform(param, lower=100, upper=500, testval=170.)
            elif param.startswith("sky"):
                v = pm.Normal(param, mu=0, sd=0.1, testval=0.)
            # Polynomial parameters
            elif param == "p0":
                v = pm.Normal("p0", mu=1, sd=0.5, testval=1.)
            elif param in polynames:
                v = pm.Normal(param, mu=0, sd=0.5, testval=0.)
            else:
                print("Parameter not found: ". param)
            vars[param] = v
        theta = []
        for param in sed.parnames:
            theta.append(vars[param])
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("n2exp", mu=0, sd=1, testval=0.)
            s = pm.Deterministic("infl", 1. + pm.math.exp(x))
            theta.append(s)
            loglike = pb.Normal2LogLike(flux, sed, obserr=fluxerr)
        theta = tt.as_tensor_variable(theta).T

        py3loglike = pb.TheanoLogLikeInterface(loglike)
        pm.DensityDist('loglike', lambda v: py3loglike(v),
                       observed={'v': theta})
    return model

def run_emcee(flam, flamerr, sed, db, loglike="normal2", model="CvD"):
    pnames = copy.deepcopy(sed.parnames)
    if loglike == "normal2":
        pnames.append("infl")
    if loglike == "studt":
        pnames.append("nu")
    mcmc_db = os.path.join(os.getcwd(), "MCMC_{}".format(model))
    trace = load_traces(mcmc_db, pnames)
    ndim = len(pnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    priors = []
    for i, param in enumerate(pnames):
        if len(param.split("_")) == 2:
            pname, n = param.split("_")
        else:
            pname = param
        ########################################################################
        # Setting first guess and limits of models
        ########################################################################
        # Stellar population parameters
        if pname in sed.ssppars:
            vmin, vmax = sed.ssppars[pname]
        else:
            vmin = np.percentile(trace[:,i], 1)
            vmax = np.percentile(trace[:,i], 99)
        prior = stats.uniform(vmin, vmax - vmin)
        priors.append(prior.logpdf)
        pos[:, i] = prior.rvs(nwalkers)
    if loglike == "normal2":
        log_likelihood = pb.Normal2LogLike(flam, sed, obserr=flamerr)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(priors, theta)])
        if not np.isfinite(lp) or np.isnan(lp):
            return -np.inf
        return log_likelihood(theta)
    backend = emcee.backends.HDFBackend(db)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, 5000, progress=True)
    return

def load_traces(db, params):
    if not os.path.exists(db):
        return None
    ntraces = len(os.listdir(db))
    data = [np.load(os.path.join(db, _, "samples.npz")) for _ in
            os.listdir(db)]
    traces = []
    # Reading weights
    for param in params:
        if param.startswith("w_"):
            w = np.vstack([data[num]["w"] for num in range(ntraces)])
            n = param.split("_")[1]
            v = w[:, int(n) - 1]
        else:
            v = np.vstack([data[num][param] for num in range(ntraces)]
                          ).flatten()
        traces.append(v)
    traces = np.column_stack(traces)
    return traces

def plot_corner(trace, outroot, title=None, redo=False):
    global labels
    title = "" if title is None else title
    output = "{}_corner.png".format(outroot)
    if os.path.exists(output) and not redo:
        return
    N = len(trace.colnames)
    params = trace.colnames
    data = np.stack([trace[p] for p in params]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    title = [title]
    for i, param in enumerate(params):
        s = "{0}$={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$".format(
            labels[param], v[i], vuerr[i], vlerr[i])
        title.append(s)
    fig, axs = plt.subplots(N, N, figsize=(3.54, 3.5))
    grid = np.array(np.meshgrid(params, params)).reshape(2, -1).T
    for i, (p1, p2) in enumerate(grid):
        i1 = params.index(p1)
        i2 = params.index(p2)
        ax = axs[i // N, i % N]
        ax.tick_params(axis="both", which='major',
                       labelsize=4)
        if i // N < i % N:
            ax.set_visible(False)
            continue
        x = data[:,i1]
        if p1 == p2:
            sns.kdeplot(x, shade=True, ax=ax, color="C0")
        else:
            y = data[:, i2]
            sns.kdeplot(x, y, shade=True, ax=ax, cmap="Blues")
            ax.axhline(np.median(y), ls="-", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 16), ls="--", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 84), ls="--", c="k", lw=0.5)
        if i > N * (N - 1) - 1:
            ax.set_xlabel(labels[p1], size=7)
        else:
            ax.xaxis.set_ticklabels([])
        if i in np.arange(0, N * N, N)[1:]:
            ax.set_ylabel(labels[p2], size=7)
        else:
            ax.yaxis.set_ticklabels([])
        ax.axvline(np.median(x), ls="-", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 16), ls="--", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 84), ls="--", c="k", lw=0.5)
    plt.text(0.6, 0.7, "\n".join(title), transform=plt.gcf().transFigure,
             size=8)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in ["png", "pdf"]:
        output = "{}_corner.{}".format(outroot, fmt)
        plt.savefig(output, dpi=300)
    plt.close(fig)
    return

def plot_fitting(wave, flux, fluxerr, sed, traces, db, redo=True, sky=None):
    global labels
    outfig = "{}_fitting".format(db.replace(".h5", ""))
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    spec = np.zeros((len(traces), len(wave)))
    loglike = pb.NormalLogLike(flux, sed, obserr=fluxerr)
    llike = np.zeros(len(traces))
    for i in tqdm(range(len(traces)), desc="Loading spectra for plots and "
                                           "table..."):
        spec[i] = sed(traces[i])
        llike[i] = loglike(traces[i])
    skyspec = np.zeros((len(traces), len(wave)))
    if sky is not None:
        idx = [i for i,p in enumerate(sed.parnames) if p.startswith("sky")]
        skytrace = traces[:, idx]
        for i in tqdm(range(len(skytrace)), desc="Loading sky models"):
            skyspec[i] = sky(skytrace[i])
    fig = plt.figure()
    plt.plot(llike)
    plt.savefig("{}_loglike.png".format(db))
    plt.close(fig)
    summary = []
    for i, param in enumerate(sed.ssppars):
        t = traces[:,i]
        m = np.median(t)
        lowerr = m - np.percentile(t, 16)
        uperr = np.percentile(t, 84) - m
        s = "{}=${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(labels[param], m,
                                                       uperr, lowerr)
        summary.append(s)
    lw=1
    y = np.median(spec, axis=0)
    skymed = np.median(skyspec, axis=0)
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * context.fig_width, 3))
    ax = plt.subplot(axs[0])
    ax.plot(wave, flux, "-", c="0.8", lw=lw)
    ax.fill_between(wave, flux - fluxerr, flux + fluxerr, color="C0", alpha=0.7)
    ax.plot(wave, flux - skymed, "-", label="M85", lw=lw)
    ax.plot(wave, y - skymed, "-", lw=lw, label="Model")
    for c, per in zip(colors, percs):
        ax.fill_between(wave, np.percentile(spec, per, axis=0) - skymed,
                         np.percentile(spec, per + 10, axis=0) - skymed,
                        color=c)
    ax.set_ylabel("Normalized flux")
    ax.xaxis.set_ticklabels([])
    ax.text(0.03, 0.88, "   ".join(summary), transform=ax.transAxes, fontsize=6)
    plt.legend()
    ax = plt.subplot(axs[1])
    for c, per in zip(colors, percs):
        ax.fill_between(wave,
                        100 * (flux - np.percentile(spec, per, axis=0)) / flux,
                        100 * (flux - np.percentile(spec, per + 10, axis=0)) /
                        flux, color=c)
    rmse = np.std((flux - y)/flux)
    ax.plot(wave, 100 * (flux - y) / flux, "-", lw=lw, c="C1",
            label="RMSE={:.1f}\%".format(100 * rmse))
    ax.axhline(y=0, ls="--", c="k", lw=1, zorder=1000)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel("Residue (\%)")
    ax.set_ylim(-5, 5)
    plt.legend()
    plt.subplots_adjust(left=0.065, right=0.995, hspace=0.02, top=0.99,
                        bottom=0.11)
    plt.savefig("{}.png".format(outfig), dpi=250)
    return

def make_table(trace, outtab):
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


def run_paintbox(cubename, velscale=200, ssp_model="CvD", sample_emiles="all"):
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
    wave = np.exp(logwave)[20:-20]
    flux, fluxerr = spectres(wave, wave_lin, flux_interp(wave_lin),
                             spec_errs=fluxerr_interp(wave_lin))
    norm = np.median(flux)
    flux /= norm
    fluxerr /= norm
    rname = cubename.split("_")[-1].split(".")[0]
    print("Producing SED model...")
    if ssp_model == "CvD":
        sed = build_sed_CvD(wave)
    elif ssp_model == "emiles":
        sed = build_sed_model_emiles(wave, sample=sample_emiles)
    print("Build pymc3 model")
    model = make_pymc3_model(flux, sed, fluxerr=fluxerr)
    mcmc_db = os.path.join(os.getcwd(), "MCMC_{}_{}".format(ssp_model, rname))
    if not os.path.exists(mcmc_db):
        with model:
            trace = pm.sample(step=pm.Metropolis())
            pm.save_trace(trace, mcmc_db, overwrite=True)
    # Run second method using initial results from MH run
    emcee_db = os.path.join(os.getcwd(), "emcee_{}_{}.h5".format(ssp_model,
                                                                 rname))
    if not os.path.exists(emcee_db):
        print("Running EMCEE...")
        run_emcee(flux, fluxerr, sed, emcee_db)
    reader = emcee.backends.HDFBackend(emcee_db)
    samples = reader.get_chain(discard=500, flat=True, thin=100)
    emcee_traces = samples[:, :len(sed.parnames)]
    print(sed.sspcolnames)
    idx = [sed.parnames.index(p) for p in sed.sspcolnames]
    ptrace_emcee = Table(emcee_traces[:, idx], names=sed.sspcolnames)
    print("Producing corner plots...")
    plot_corner(ptrace_emcee, emcee_db.replace(".h5", ""),
                title="{} {}".format(galaxy, rname),
                redo=True)
    print("Producing fitting figure...")
    plot_fitting(wave, flux, fluxerr, sed, emcee_traces, emcee_db,
                 redo=False)
    print("Making summary table...")
    outtab = os.path.join(emcee_db.replace(".h5", "_results.fits"))
    summary_pars = sed.parnames
    idx = [sed.parnames.index(p) for p in summary_pars]
    summary_trace = Table(emcee_traces[:, idx], names=summary_pars)
    make_table(summary_trace, outtab)

if __name__ == "__main__":
    ssp_model = "CvD"
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]",
              "Na": "[Na/Fe]" if ssp_model == "emiles" else "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]"}
    wdir = os.path.join(context.home, "elliot")
    sample = os.listdir(wdir)
    for galaxy in sample:
        galdir = os.path.join(wdir, galaxy)
        os.chdir(galdir)
        cubenames = [_ for _ in os.listdir(galdir) if _.endswith(".fits")]
        for cubename in cubenames:
            run_paintbox(cubename, ssp_model=ssp_model)
        break