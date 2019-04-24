# -*- coding: utf-8 -*-
""" 

Created on 23/10/18

Author : Carlos Eduardo Barbosa

Produces maps of kinematics.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.table import Table, hstack
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import context

def plot_from_voronoi(vorfile, fields, outdir, lims=None, labels=None,
             cmaps=None, wcs=None):
    """ Produces maps of properties in a given table using a Voronoi image."""
    lims = [[None, None]] * len(fields) if lims is None else lims
    labels = [_.replace("_", "") for _ in fields] if labels is None else labels
    cmaps = ["Spectral_r"] * len(fields) if cmaps is None else cmaps
    vorimg = fits.getdata(vorfile, ext=0)
    table = Table.read(vorfile)
    h = fits.getheader(vorfile, ext=0)
    wcs = WCS(h)
    ydim, xdim = vorimg.shape
    xpix = np.arange(xdim) + 1
    ypix = np.arange(ydim) + 1
    xx, yy = np.meshgrid(xpix, ypix)
    coords = np.column_stack((xx.flatten(), yy.flatten()))
    ra, dec = wcs.all_pix2world(coords, 1).T
    ra = ra.reshape(vorimg.shape)
    dec = dec.reshape(vorimg.shape)
    ra0, dec0 = wcs.all_pix2world([[xdim / 2, ydim/2]], 1).T
    ra = (ra - ra0) * 3600.
    dec = (dec - dec0) * 3600.
    for i, field in enumerate(fields):
        img = np.zeros(vorimg.shape, dtype=np.float32) * np.nan
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.minorticks_on()
        vmin = lims[i][0]
        vmax = lims[i][1]
        vmin = np.percentile(table[field], 10) if vmin is None else vmin
        vmax = np.percentile(table[field], 90) if vmax is None else vmax
        for t in table:
            idx = np.where(vorimg==t["binnum"])
            img[idx] = t[field]
        if wcs is None:
            im = ax.imshow(img, origin="bottom", vmin=vmin, vmax=vmax,
                           cmap=cmaps[i])
        else:
            im = ax.pcolormesh(ra, dec, img, vmin=vmin, vmax=vmax,
                           cmap=cmaps[i])
        ax.set_xlabel("$\Delta$RA (arcsec)")
        ax.set_ylabel("$\Delta$DEC (arcsec)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(labels[i])
        plt.subplots_adjust(left=0.11, top=0.98, bottom=0.12, right=0.88)
        plt.savefig(os.path.join(outdir, "map_{}.png".format(field)), dpi=300)
    return


if __name__ == "__main__":
    targetSN = 40
    w1 = 8500
    w2 = 13500
    velscale = 20
    imgfile = os.path.join(context.data_dir, "M85_combined_cubeImg_2.fits")
    h = fits.getheader(imgfile)
    wcs = WCS(h)
    kinfile = os.path.join(context.data_dir, "ppxf_vel{}_sn{}_w{}_{}.fits"
                          "".format(int(velscale), targetSN, w1, w2))
    kintable = Table.read(kinfile)
    vorfile = os.path.join(context.data_dir, "voronoi_sn{}.fits".format(
                            targetSN))
    vortable = Table.read(vorfile)
    vorimg = fits.getdata(vorfile, ext=0)
    imax = np.minimum(len(vortable), len(kintable))
    table = hstack([vortable[:imax], kintable[:imax]])
    fields = ["v", "sigma", "h3", "h4"]
    labels = ["V (km/s)", "$\sigma$ (km/s)", "$h_3$", "$h_4$"]
    outdir = os.path.join(context.data_dir, "plots", "ppxf_sn{}".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    lims = [[780, 950], [100, 250]] + 2 * [[None, None]]
    plot_map(table, vorimg, fields, outdir, labels=labels, wcs=wcs,
             lims=lims)
