# -*- coding: utf-8 -*-
"""

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Run paintbox in the central part of M85 using WIFIS data.

"""

import os
import shutil
import yaml
from datetime import datetime
import subprocess
import copy

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table
from astroquery.vizier import Vizier
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

import context
from paintbox import paintbox as pb

def data_reduction_m85(redo=False):
    output = os.path.join(os.getcwd(), "M85_1D.fits")
    if os.path.exists(output) and not redo:
        return
    datacube = os.path.join(context.data_dir, "M85_combined_cube_2.fits")
    datacube_err = os.path.join(context.data_dir, "M85_unc_cube.fits")
    datacube_t = os.path.join(context.data_dir,
                              "HIP56736_combined_cube_1.fits")
    hdr = fits.getheader(datacube)
    wave = ((np.arange(hdr['NAXIS3']) + 1 - hdr['CRPIX3']) * hdr['CDELT3'] + \
            hdr['CRVAL3']) * u.m
    hdr_t = fits.getheader(datacube_t)
    wave_t = ((np.arange(hdr_t['NAXIS3']) + 1 - hdr_t['CRPIX3']) * hdr_t[
        'CDELT3'] + hdr_t['CRVAL3']) * u.m
    wave = wave.to(u.micrometer).value
    wave_t = wave_t.to(u.micrometer).value
    spec1d = extract_spectrum(datacube)
    spec1d_e = extract_spectrum(datacube_err, errcube=True)
    spec1d_t = extract_spectrum(datacube_t)
    # Select wavelength to work
    w = np.where(np.isfinite(spec1d * spec1d_e), wave, np.nan)
    idx = np.where((wave >= np.nanmin(w)) & (wave <= np.nanmax(w)))[0]
    wave = wave[idx]
    spec1d = spec1d[idx]
    spec1d_e = spec1d_e[idx]
    spec1d_t = spectres(wave, wave_t, spec1d_t)
    table = Table([wave, spec1d, spec1d_e, np.zeros_like(wave)],
                  names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    table.write("M85_1D.fits", overwrite=True)
    table = Table([wave, spec1d_t, np.zeros_like(wave), np.zeros_like(wave)],
                  names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    table.write("HIP56736_1D.fits", overwrite=True)
    return

def extract_spectrum(datacube, r=None, x0=44, y0=19, errcube=False):
    r = 2 * u.arcsec if r is None else r
    data = fits.getdata(datacube)
    if len(data.shape) > 3:
        data = data[0]
    zdim, ydim, xdim = data.shape
    wcs = WCS(datacube)
    ps = (proj_plane_pixel_scales(wcs)[:2].mean() * u.degree).to(u.arcsec)
    aper = (r / ps).value
    X, Y = np.meshgrid(np.arange(xdim), np.arange(ydim))
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    idx = np.where(R <= aper)
    specs = data[:, idx[0], idx[1]]
    if errcube:
        spec1D = np.sqrt(np.nanmean(specs**2, axis=1))
    else:
        spec1D = np.nanmean(specs, axis=1)
    return spec1D

def run_molecfit_m85(redo=False):
    output = os.path.join(os.getcwd(), "molecfit/M85_1D_TAC.fits")
    if os.path.exists(output) and not redo:
        return
    stdimg = os.path.join(context.data_dir, "HIP56736_combined_cubeImg_1.fits")
    stdheader = fits.getheader(stdimg)
    jd = float(stdheader["BARY_JD"])
    mjd = int(jd - 2400000.5)
    utc = datetime.strptime("12:19:32.5", "%H:%M:%S.%f")
    utc_seconds = utc.hour * 3600 + utc.minute * 60 + utc.second
    molecfit_params = {"user_workdir": os.getcwd(),
                       "filename": "HIP56736_1D.fits",
                       "listname": "filelist.txt",
                       "obsdate": mjd, "utc": utc_seconds,
                       "telalt": stdheader["ELEVATIO"]}
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "molecfit")
    # Copying default parameter files
    for fname in os.listdir(config_path):
        shutil.copy(os.path.join(config_path, fname), os.getcwd())
    config_file = os.path.join(os.getcwd(), "wifis_zJ.par")
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in molecfit_params.keys():
        if key in config.keys():
            config[key] = molecfit_params[key]
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    subprocess.run(["bash", "/home/kadu/molecfit/bin/molecfit", config_file])
    subprocess.run(["bash", "/home/kadu/molecfit/bin/calctrans", config_file])
    # Changing columns to apply corrfilelist
    config["columns"] = "WAVE FLUX FLUX_ERR MASK"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    subprocess.run(["bash", "/home/kadu/molecfit/bin/corrfilelist", config_file])

def flux_calibration_m85(redo=False):
    output = os.path.join(os.getcwd(), "M85_sci.fits")
    if os.path.exists(output) and not redo:
        return
    # Make flux calibration
    two_mass = Vizier(columns=["*", "+_r"])
    std2mass = two_mass.query_object("HIP56736", catalog="II/246")[0][0]
    ref2mass = two_mass.query_object("Vega", catalog="II/246")[0][0]
    dmag = std2mass["Jmag"] - ref2mass["Jmag"]
    observed = Table.read("HIP56736_1D_TAC.fits")
    template = Table.read(os.path.join(context.data_dir,
                                       "rieke2008/table7.fits"))
    template.rename_column("lambda", "WAVE")
    template.rename_column("Vega", "FLUX")
    # Cropping template in wavelength
    idx = np.where((template["WAVE"] >= observed["WAVE"].min()) &
                    (template["WAVE"] <= observed["WAVE"].max()))
    template = template[idx]
    ############################################################################
    # Scaling the flux of Vega to that of the standard star
    stdflux = template["FLUX"] * np.power(10, -0.4 * dmag)
    # Determining the sensitivity function
    sensfun = calc_sensitivity_function(observed["WAVE"], observed["tacflux"],
                            template["WAVE"], stdflux)
    # Applying sensitivity function to galaxy spectra
    table = Table.read("M85_1D_TAC.fits")
    wave = table["WAVE"]
    newflux = table["tacflux"].data * sensfun(wave)
    newfluxerr = table["tacdflux"].data * sensfun(wave)
    newtable = Table([wave, newflux, newfluxerr], names=["WAVE", "FLUX",
                                                         "FLUX_ERR"])
    newtable.write(output, overwrite=True)
    return

def calc_sensitivity_function(owave, oflux, twave, tflux, order=30):
    """ Calculates the sensitivity function using a polynomial approximation.
    """

    # Setting the appropriate wavelength regime
    wmin = np.maximum(owave[1], twave[1])
    wmax = np.minimum(owave[-2],twave[-2])
    dw = 0.1 * np.minimum(owave[1] - owave[0], twave[1] - twave[0])
    wave = np.arange(wmin, wmax, dw)
    # Rebinning and normalizing spectra
    oflux = spectres(wave, owave, oflux)
    tflux = spectres(wave, twave, tflux)
    sens = np.poly1d(np.polyfit(wave, tflux / oflux, order))
    return sens

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
    sky = pb.Polynomial(wave, 0)
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
    with model:
        theta = []
        for param in sed.parnames:
            # Stellar population parameters
            if param in sed.ssppars:
                vmin, vmax = sed.ssppars[param]
                vinit = float(0.5 * (vmin + vmax))
                v = pm.Uniform(param, lower=float(vmin), upper=float(vmax),
                               testval=vinit)
                theta.append(v)
            # Dust attenuation parameters
            elif param == "Av":
                Av = pm.Exponential("Av", lam=1 / 0.4, testval=0.1)
                theta.append(Av)
            elif param == "Rv":
                BNormal = pm.Bound(pm.Normal, lower=0)
                Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
                theta.append(Rv)
            elif param == "V":
                # Stellar kinematics
                V = pm.Normal("V", mu=729., sd=50., testval=729)
                theta.append(V)
            elif param == "sigma":
                sigma = pm.Uniform(param, lower=100, upper=500, testval=170.)
                theta.append(sigma)
            elif param.startswith("sky"):
                sky = pm.Normal(param, mu=0, sd=0.1, testval=0.)
                theta.append(sky)
            # Polynomial parameters
            elif param == "p0":
                p0 = pm.Normal("p0", mu=1, sd=0.1, testval=1.)
                theta.append(p0)
            elif param in polynames:
                pn = pm.Normal(param, mu=0, sd=0.01, testval=0.)
                theta.append(pn)
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("x", mu=0, sd=1, testval=0.)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = pb.TheanoLogLikeInterface(flux, sed, loglike=loglike,
                                          obserr=fluxerr)
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
    return model

def run_emcee(flam, flamerr, sed, db, loglike="normal2"):
    pnames = copy.deepcopy(sed.parnames)
    if loglike == "normal2":
        pnames.append("S")
    if loglike == "studt":
        pnames.append("nu")
    mcmc_db = os.path.join(os.getcwd(), "MCMC")
    trace = load_traces(mcmc_db, pnames)
    ndim = len(pnames)
    nwalkers = 2 * ndim
    polynames = ["p{}".format(i+1) for i in range(sed.porder)]
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
    sampler.run_mcmc(pos, 1000, progress=True)
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
    title = "" if title is None else title
    output = "{}_corner.png".format(outroot)
    if os.path.exists(output) and not redo:
        return
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
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
    sspdict = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age",
               "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    summary = []
    for i, param in enumerate(sed.ssppars):
        t = traces[:,i]
        m = np.median(t)
        lowerr = m - np.percentile(t, 16)
        uperr = np.percentile(t, 84) - m
        s = "{}=${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(sspdict[param], m,
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
    tab = Table()
    for i, param in enumerate(trace.colnames):
        tab[param] = [round(v[i], 5)]
        tab["{}_lerr".format(param)] = [round(vlerr[i], 5)]
        tab["{}_uerr".format(param)] = [round(vuerr[i], 5)]
    tab.write(outtab, overwrite=True)
    return tab


def run_paintbox_m85(velscale=200, sample="all"):
    # Read first spectrum to set the dispersion
    data = Table.read("M85_sci.fits")
    flux = data["FLUX"].data
    fluxerr = data["FLUX_ERR"].data
    wave_lin = (data["WAVE"] * u.micrometer).to(u.AA).value
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]],
                                           data["FLUX"], velscale=velscale)
    wave = np.exp(logwave)[1:-1]
    flux, fluxerr = spectres(wave, wave_lin, flux, spec_errs=fluxerr)
    norm = np.median(flux)
    flux /= norm
    fluxerr /= norm
    print("Producing SED model...")
    sed = build_sed_model_emiles(wave, sample=sample)
    print("Build pymc3 model")
    model = make_pymc3_model(flux, sed, fluxerr=fluxerr)
    mcmc_db = os.path.join(os.getcwd(), "MCMC")
    if not os.path.exists(mcmc_db):
        with model:
            trace = pm.sample(draws=500, tune=500, step=pm.Metropolis())
            pm.save_trace(trace, mcmc_db, overwrite=True)
    # Run second method using initial results from MH run
    emcee_db = os.path.join(os.getcwd(), "M85_emcee.h5")
    if not os.path.exists(emcee_db):
        print("Running EMCEE...")
        run_emcee(flux, fluxerr, sed, emcee_db)
    reader = emcee.backends.HDFBackend(emcee_db)
    samples = reader.get_chain(discard=500, flat=True, thin=100)
    emcee_traces = samples[:, :len(sed.parnames)]
    idx = [sed.parnames.index(p) for p in sed.sspcolnames]
    ptrace_emcee = Table(emcee_traces[:, idx], names=sed.sspcolnames)
    print("Producing corner plots...")
    plot_corner(ptrace_emcee, emcee_db, title="M85", redo=True)
    print("Producing fitting figure...")
    plot_fitting(wave, flux, fluxerr, sed, emcee_traces, emcee_db,
                 redo=True)
    print("Making summary table...")
    outtab = os.path.join(emcee_db.replace(".h5", "_results.fits"))
    summary_pars = sed.sspcolnames + ["Av", "V", "sigma"]
    idx = [sed.parnames.index(p) for p in summary_pars]
    summary_trace = Table(emcee_traces[:, idx], names=summary_pars)
    make_table(summary_trace, outtab)

if __name__ == "__main__":
    wdir = os.path.join(context.home, "paintbox")
    run_paintbox_m85()