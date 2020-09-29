# -*- coding: utf-8 -*-
"""

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Run paintbox in the central part of M85 using WIFIS data.

"""

import os
import getpass

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from matplotlib import cm
from spectres import spectres
import ppxf.ppxf_util as util
from tqdm import tqdm
import seaborn as sns
import emcee
from scipy import stats
from scipy.interpolate import interp1d

import context
import paintbox as pb
from paintbox.interfaces import TheanoLogLikeInterface

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

def build_sed_CvD(wave, velscale=200, porder=45, elements=None, V=0):
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
    priors = {}
    limits = {}
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        limits[param] = (vmin, vmax)
        priors[param] = stats.uniform(loc=vmin, scale=vmax-vmin)
    ssp = pb.ParametricModel(twave, params, templates)
    for element in elements:
        elem_file = os.path.join(wdir, "C18_rfs_wifis_{}.fits".format(element))
        rfdata = fits.getdata(elem_file, ext=0)
        rfpar = Table.read(elem_file, hdu=1)
        vmin, vmax = rfpar[element].min(), rfpar[element].max()
        limits[element] = (vmin, vmax)
        priors[element] = stats.uniform(loc=vmin, scale=vmax-vmin)
        ewave = Table.read(elem_file, hdu=2)["wave"].data
        rf = pb.ParametricModel(ewave, rfpar, rfdata)
        ssp = ssp * rf
    # Adding extinction to the stellar populations
    stars = pb.Resample(wave, pb.LOSVDConv(ssp, velscale=velscale))
    priors["V"] = stats.norm(loc=V, scale=100)
    priors["sigma"] = stats.uniform(loc=100, scale=200)
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    for p in poly.parnames:
        if p == "p0":
            mu, sd = 1, 0.3
            a, b = (0 - mu) / sd, (np.infty - mu) / sd
            priors[p] = stats.truncnorm(a, b, mu, sd)
        else:
            priors[p] = stats.norm(0, 0.02)
    # Using Polynomial to make sky model
    sky = pb.Polynomial(wave, 0)
    sky.parnames = [_.replace("p", "sky") for _ in sky.parnames]
    priors["sky0"] = stats.norm(0, 0.1)
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

def plot_corner(trace, outroot, title=None, redo=False):
    global labels
    title = "" if title is None else title
    output = "{}.png".format(outroot)
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
    for i, (p1, p2) in enumerate(tqdm(grid, desc="producing corner plots")):
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
            sns.kdeplot(x, shade=True, ax=ax, color="tab:blue")
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
    plt.text(0.6, 0.55, "\n".join(title), transform=plt.gcf().transFigure,
             size=8)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in tqdm(["png", "pdf"], desc="Saving to disk"):
        output = "{}.{}".format(outroot, fmt)
        plt.savefig(output, dpi=200)
    plt.close(fig)
    return

def plot_fitting(wave, flux, fluxerr, sed, traces, db, redo=True, sky=None,
                 norm=1, unit_norm=1, lw=1, name=None, ylabel=None,
                 reslabel=None):
    global labels
    ylabel = "$f_\lambda$ ($10^{{-{0}}}$ " \
             "erg cm$^{{-2}}$ s$^{{-1}}$ \\r{{A}}$^{{-1}}$)".format(unit_norm) \
             if ylabel is None else ylabel
    reslabel = "$\Delta f_\lambda$" if reslabel is None else reslabel
    outfig = "{}_fitting".format(db.replace(".h5", ""))
    name = "Observed" if name is None else name
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    percs = np.array([2.2, 15.8, 84.1, 97.8])
    fracs = np.array([0.3, 0.6, 0.3])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    models = np.zeros((len(traces), len(wave)))
    loglike = pb.NormalLogLike(flux, sed, obserr=fluxerr)
    llike = np.zeros(len(traces))
    for i in tqdm(range(len(traces)), desc="Loading spectra for plots and "
                                           "table..."):
        models[i] = sed(traces[i])
        llike[i] = loglike(traces[i])
    outmodels = db.replace(".h5", "_seds.fits")
    hdu0 = fits.PrimaryHDU(models)
    hdu1 = fits.ImageHDU(wave)
    hdus = [hdu0, hdu1]
    skyspec = np.zeros((len(traces), len(wave)))
    if sky is not None:
        idx = [i for i,p in enumerate(sed.parnames) if p.startswith("sky")]
        skytrace = traces[:, idx]
        for i in tqdm(range(len(skytrace)), desc="Loading sky models"):
            skyspec[i] = sky(skytrace[i])
        hdu2 = fits.ImageHDU(skyspec)
        hdus.append(hdu2)
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(outmodels, overwrite=True)
    fig = plt.figure()
    plt.plot(llike)
    plt.savefig("{}.png".format(db.replace("_corner", "_loglike")))
    plt.close(fig)
    summary = []
    for i, param in enumerate(sed.sspcolnames + ["V", "sigma"]):
        if (i % 5 == 0) and (i>0):
            summary.append("\n")
        t = traces[:,i]
        m = np.median(t)
        lowerr = m - np.percentile(t, 16)
        uperr = np.percentile(t, 84) - m
        s = "{}=${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(labels[param], m,
                                                       uperr, lowerr)
        summary.append(s)
    ############################################################################
    # Normalizing spectra for plots
    skyspec *= norm * np.power(10., unit_norm)
    models *= norm * np.power(10., unit_norm)
    flux *= norm * np.power(10., unit_norm)
    fluxerr *= norm * np.power(10., unit_norm)
    ############################################################################
    skymed = np.median(skyspec, axis=0)
    bestfit = np.median(models, axis=0)
    flux0 = flux - skymed
    # Starting figure
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * context.fig_width, 3))
    ax = plt.subplot(axs[0])
    ax.fill_between(wave, flux + fluxerr, flux - fluxerr, color="0.8")
    ax.fill_between(wave, flux0 + fluxerr, flux0 - fluxerr,
                    "-", label=name, color="tab:blue")
    for i in [0, 2, 1]:
        c = colors[i]
        per = percs[i]
        label = "Model" if i == 1 else None
        ax.fill_between(wave, np.percentile(models, per, axis=0) - skymed,
                         np.percentile(models, percs[i+1], axis=0) - skymed,
                        color=c, label=label, lw=lw)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels([])
    ax.text(0.1, 0.7, "   ".join(summary), transform=ax.transAxes, fontsize=7)
    plt.legend(loc=7)
    ylim = ax.get_ylim()
    ax.set_ylim(None, 1.1 * ylim[1])
    # Residual plot
    ax = plt.subplot(axs[1])
    p = flux - bestfit
    sigma_mad = 1.4826 * np.median(np.abs(p - np.median(p)))
    ax.fill_between(wave, skymed - fluxerr, skymed + fluxerr, color="0.8")
    ax.fill_between(wave, fluxerr, -fluxerr, "-", color="tab:blue")
    sigma_per = sigma_mad/np.median(flux)
    for i in [0, 2, 1]:
        c = colors[i]
        per = percs[i]
        label = "$\sigma_{{MAD}}$={:.1f}%".format(sigma_per) if i==1 \
                 else None
        ax.fill_between(wave, np.percentile(models, per, axis=0) - skymed -
                        flux0,
                         np.percentile(models, percs[i+1], axis=0) - skymed -
                        flux0,
                        color=c, lw=lw, label=label)
    ax.set_ylim(-5 * sigma_mad, 5 * sigma_mad)
    ax.axhline(y=0, ls="--", c="k", lw=1, zorder=1000)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel(reslabel)
    plt.legend(loc=1, framealpha=1)
    plt.subplots_adjust(left=0.07, right=0.995, hspace=0.02, top=0.99,
                        bottom=0.11)
    fig.align_ylabels(axs)
    plt.savefig("{}.png".format(outfig), dpi=250)
    plt.close()
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

def run_paintbox(galaxy, radius, V, velscale=200, ssp_model="CvD",
                 sample_emiles="all", loglike="normal2", elements=None,
                 nsteps=4000, postprocessing=False):
    cubename = "{}_combined_cube_1_telluricreduced_{}_{}.fits".format(
                galaxy, date[galaxy], radius)
    name = "{} {}".format(galaxy, radius)
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
    wave = np.exp(logwave)[20:-20]
    flux, fluxerr = spectres(wave, wave_lin, flux_interp(wave_lin),
                             spec_errs=fluxerr_interp(wave_lin))
    ############################################################################
    # Inflationary term for testing with Student's T loglike
    if loglike == "studt":
        factor = 3. if radius=="R1" else 2.
        fluxerr *= factor
    else:
        factor = 1.
    ############################################################################
    # Normalizing the data
    norm = np.median(flux)
    flux /= norm
    fluxerr /= norm
    radius = cubename.split("_")[-1].split(".")[0]
    print("Producing SED model...")
    if ssp_model == "CvD":
        sed, priors = build_sed_CvD(wave, elements=elements, V=V)
    elif ssp_model == "emiles":
        sed = build_sed_model_emiles(wave, sample=sample_emiles)
    logp = pb.StudTLogLike(flux, sed, obserr=fluxerr)
    priors["nu"] = stats.uniform(loc=2.01, scale=8)
    #
    outdb = os.path.join(os.getcwd(), "{}_{}_{}_{}_{}.h5".format(
                        galaxy, radius, ssp_model, loglike, elements_str))
    corner_name = outdb.replace(".h5", "_corner")
    if not os.path.exists(outdb):
        print("Running emcee...")
        run_sampler(logp, priors, outdb, nsteps=nsteps)
    ############################################################################
    # Only make postprocessing locally, not @ alphacrucis
    if not postprocessing:
        return
    reader = emcee.backends.HDFBackend(outdb)
    trace = reader.get_chain(discard=int(.9 * nsteps), flat=True, thin=80)
    print("Using trace with {} samples".format(len(trace)))
    if ssp_model == "CvD":
        corner_pars = ['Z', 'Age', 'x1', 'x2', 'Na', "Fe", 'Ca', "K"]
    elif ssp_model == "emiles":
        corner_pars = ['Z', 'T', 'x1', 'x2', 'Na', "Fe", 'Ca', "K"]
    idx = [sed.parnames.index(p) for p in corner_pars]
    ptrace_emcee = Table(trace[:, idx], names=corner_pars)
    print("Producing corner plots...")
    plot_corner(ptrace_emcee, corner_name,
                title="{} {}".format(galaxy, radius),
                redo=False)
    print("Producing fitting figure...")
    plot_fitting(wave, flux, fluxerr / factor, sed, trace, outdb,
                 redo=True, norm=norm, name=name, ylabel="Flux",
                 reslabel="Residuals")
    print("Making summary table...")
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    summary_trace = Table(trace, names=logp.parnames)
    make_table(summary_trace, outtab)

if __name__ == "__main__":
    postprocessing = True if getpass.getuser() == "kadu" else False
    ssp_model = "CvD"
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]", "Age": "Age (Gyr)",
              "Na": "[Na/Fe]" if ssp_model == "emiles" else "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]",
              "V": "$V_*$ (km/s)", "sigma": "$\sigma_*$ (km/s)"}
    wdir = os.path.join(context.home, "elliot")
    sample = ["M85", "NGC5557"]
    date = {"M85": "20200528", "NGC5557": "20200709"}
    V = {"M85": 729, "NGC5557": 3219}
    elements = ["Na", "Fe", "Ca", "K"]
    for galaxy in sample[::-1]:
        galdir = os.path.join(wdir, galaxy)
        os.chdir(galdir)
        for radius in ["R1", "R2"]:
            run_paintbox(galaxy, radius, V[galaxy], ssp_model=ssp_model,
                         loglike="studt", elements=elements,
                         postprocessing=postprocessing)