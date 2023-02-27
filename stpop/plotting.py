import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
from astropy.table import Table, vstack
import paintbox as pb
from tqdm import tqdm
import seaborn as sns
import os
import context

def plot_fitting(wave, flux, fluxerr, sed, traces, tracetable, db, regions, 
                 line_name, redo=True, linefit = False, sky=None,
                 norm=1, unit_norm=1, lw=1, name=None, ylabel=None,
                 reslabel=None, liketype = 'studt'):
                 
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

    c = 299792.458
    argvsyst = np.where(np.array(sed.parnames) == 'Vsyst')[0]
    vsyst_med = np.percentile(traces[:,argvsyst], 50)
    v_c = vsyst_med/c
    z_factor = np.sqrt((1+v_c)/(1-v_c))

    models = np.zeros((len(traces), len(wave)))
    if liketype == 'studt':
        loglike = pb.StudTLogLike(flux, sed, obserr=fluxerr)
    else:
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
    print_cols = sed.sspcolnames + ["V", "sigma", "alpha_Ks", "M2L_Ks"]
    for i, param in enumerate(print_cols):
        if (i % 5 == 0) and (i>0):
            summary.append("\n")
        if param == 'V':
            t = np.array(tracetable['Vsyst'])
        else:
            t = np.array(tracetable[param])
        m = np.median(t)
        lowerr = m - np.percentile(t, 16)
        uperr = np.percentile(t, 84) - m
        #mask = tracetable['param'] == param
        #m = float(tracetable[mask]['median'])
        #lowerr = float(tracetable[mask]['lerr'])
        #uperr = float(tracetable[mask]['uerr'])
        s = "{}=${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(context.labels[param], m,
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
    fig = plt.figure(figsize=(2 * context.fig_width, 3))
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2, 1])
    ax = fig.add_subplot(gs[0,0])
    ax.fill_between(wave, flux + fluxerr, flux - fluxerr, color="0.8")
    ax.fill_between(wave, flux0 + fluxerr, flux0 - fluxerr,
                    label=name, color="tab:blue")
    for i in [0, 2, 1]:
        c = colors[i]
        per = percs[i]
        label = "Model" if i == 1 else None
        ax.fill_between(wave, np.percentile(models, per, axis=0) - skymed,
                         np.percentile(models, percs[i+1], axis=0) - skymed,
                         color=c, label=label, lw=lw)
    for region in regions:
        ax.axvspan(region[0],region[1],facecolor='k',alpha=0.2)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels([])
    ax.text(0.1, 0.7, "    ".join(summary), transform=ax.transAxes, fontsize=7)
    plt.legend(loc=7)
    ylim = ax.get_ylim()
    ax.set_ylim(None, 1.1 * ylim[1])
    # Residual plot
    ax = fig.add_subplot(gs[1, 0])
    p = flux - bestfit
    sigma_mad = 1.4826 * np.median(np.abs(p - np.median(p)))
    y0 = np.median(flux)
    y1, y2 = skymed - fluxerr, skymed + fluxerr
    ax.fill_between(wave, 100 * y1 / y0, 100 * y2 / y0, color="0.8")
    y1, y2 = fluxerr, - fluxerr
    ax.fill_between(wave, 100 * y1 / y0, 100 * y2 / y0, color="tab:blue")
    sigma_per = sigma_mad / y0 * 100
    for i in [0, 2, 1]:
        c = colors[i]
        per = percs[i]
        label = "$\sigma_{{\\rm MAD}}$={:.1f}\%".format(sigma_per) if i==1 \
                 else None
        y1 = np.percentile(models, per, axis=0) - skymed - flux0
        y2 = np.percentile(models, percs[i + 1], axis=0) - skymed - flux0
        ax.fill_between(wave, 100 * y1 / y0, 100 * y2 / y0,
                        color=c, lw=lw, label=label)
    ax.set_ylim(-6 * sigma_per, 6 * sigma_per)
    ax.axhline(y=0, ls="--", c="k", lw=1, zorder=1000)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel(reslabel)
    for region in regions:
        ax.axvspan(region[0],region[1],facecolor='k',alpha=0.2)
    plt.legend(loc=1, framealpha=1)
    plt.subplots_adjust(left=0.07, right=0.995, hspace=0.02, top=0.99,
                        bottom=0.11)
    # fig.align_ylabels(gs)
    plt.savefig("{}.png".format(outfig), dpi=250)
    plt.close()

    if linefit:
        # linefit figure
        fig, axes = plt.subplots(2,5, figsize=(10,4))
        axes = axes.flatten()
        for i in range(len(line_name)):
            plotwave = np.where(np.logical_and(wave >= regions[i][0]*z_factor,
                                    wave <= regions[i][1]*z_factor))[0]
            axes[i].plot(wave[plotwave], flux[plotwave]-skymed[plotwave], color='black')
            axes[i].plot(wave[plotwave], np.percentile(models, 50, axis=0)[plotwave]
                        - skymed[plotwave], color='orange')
            axes[i].fill_between(wave[plotwave], np.percentile(models, 16, axis=0)[plotwave]-
                        skymed[plotwave], np.percentile(models, 84, axis=0)[plotwave]-
                        skymed[plotwave],
                        alpha=0.3, color='orange')
            axes[i].set_title(line_name[i])
            #ylim = axes[i].get_ylim()
            #axes[i].set_ylim(None, 1.1 * ylim[1])

        axes[0].set_ylabel(ylabel)
        #ax.text(0.1, 0.7, "    ".join(summary), transform=ax.transAxes, fontsize=7)
        plt.legend(loc=7)
        # fig.align_ylabels(gs)
        outfig = "{}_linefitting".format(db.replace(".h5", ""))
        plt.savefig("{}.png".format(outfig), dpi=250)
        plt.close()

    return

def plot_corner(trace, parnames, outroot, title=None, redo=False):

    # Check to see if the items inside parnames exist in the model
    new_parnames = []
    for param in parnames:
        if param in trace.colnames:
            new_parnames.append(param)
    parnames = new_parnames

    title = "" if title is None else title
    output = "{}.png".format(outroot)
    if os.path.exists(output) and not redo:
        return
    N = len(parnames)
    data = np.stack([trace[p] for p in parnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    title = [title]
    for i, param in enumerate(parnames):
        s = "{0}$={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$".format(
            context.labels[param], v[i], vuerr[i], vlerr[i])
        title.append(s)
    grid = np.array(np.meshgrid(parnames, parnames)).reshape(2, -1).T
    fig = plt.figure(figsize=(3.54, 3.5))
    gs = fig.add_gridspec(ncols=N, nrows=N)
    for i, (p1, p2) in enumerate(grid):
        i1 = parnames.index(p1)
        i2 = parnames.index(p2)
        ax = fig.add_subplot(gs[i // N, i % N])
        # ax = axs[i // N, i % N]
        ax.tick_params(axis="both", which='major',
                       labelsize=4)
        if i // N < i % N:
            ax.set_visible(False)
            continue
        x = data[:,i1]
        if p1 == p2:
            hist = sns.kdeplot(x, shade=True, ax=ax, color="tab:blue",
                            legend=False)
            hist.set(ylabel=None)
        else:
            y = data[:, i2]
            sns.kdeplot(x, y, shade=True, ax=ax, cmap="Blues")
            ax.axhline(np.median(y), ls="-", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 16), ls="--", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 84), ls="--", c="k", lw=0.5)
        if i > N * (N - 1) - 1:
            ax.set_xlabel(context.labels[p1], size=7)
        else:
            ax.xaxis.set_ticklabels([])
        if i in np.arange(0, N * N, N)[1:]:
            ax.set_ylabel(context.labels[p2], size=7)
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

def plot_linefit(wave, flux, fluxerr, sed, traces, db, regions, line_name,
                 redo=True, sky=None,
                 norm=1, unit_norm=1, lw=1, name=None, ylabel=None,
                 reslabel=None, liketype = 'studt'):

    percs = np.array([2.2, 15.8, 84.1, 97.8])
    models = np.zeros((len(traces), len(wave)))
    if liketype == 'studt':
        loglike = pb.StudTLogLike(flux, sed, obserr=fluxerr)
    else:
        loglike = pb.NormalLogLike(flux, sed, obserr=fluxerr)
    llike = np.zeros(len(traces))
    for i in tqdm(range(len(traces)), desc="Loading spectra for plots and "
                                           "table..."):
        models[i] = sed(traces[i])
        llike[i] = loglike(traces[i])

    skyspec = np.zeros((len(traces), len(wave)))
    if sky is not None:
        idx = [i for i,p in enumerate(sed.parnames) if p.startswith("sky")]
        skytrace = traces[:, idx]
        for i in tqdm(range(len(skytrace)), desc="Loading sky models"):
            skyspec[i] = sky(skytrace[i])
 
    c = 299792.458
    argvsyst = np.where(np.array(sed.parnames) == 'Vsyst')[0]
    vsyst_med = np.percentile(traces[:,argvsyst], 50)
    v_c = vsyst_med/c
    z_factor = np.sqrt((1+v_c)/(1-v_c))

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

    fig, axes = plt.subplots(2,5, figsize=(10,4))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        plotwave = np.where(np.logical_and(wave >= regions[i][0]*z_factor,
                                wave <= regions[i][1]*z_factor))[0]
        ax.plot(wave[plotwave], flux[plotwave]-skymed[plotwave], color='black')
        ax.plot(wave[plotwave], np.percentile(models, 50, axis=0)[plotwave]
                    - skymed[plotwave], color='orange')
        ax.fill_between(wave[plotwave], np.percentile(models, 16, axis=0)[plotwave]-
                    skymed[plotwave], np.percentile(models, 84, axis=0)[plotwave]-
                    skymed[plotwave],
                    alpha=0.3, color='orange')
        ax.set_title(line_name[i])

    axes[0].set_ylabel(ylabel)
    #ax.text(0.1, 0.7, "    ".join(summary), transform=ax.transAxes, fontsize=7)
    plt.legend(loc=7)
    ylim = ax.get_ylim()
    ax.set_ylim(None, 1.1 * ylim[1])
    # fig.align_ylabels(gs)
    outfig = "{}_linefitting".format(db.replace(".h5", ""))
    plt.savefig("{}.png".format(outfig), dpi=250)
    plt.close()
    return