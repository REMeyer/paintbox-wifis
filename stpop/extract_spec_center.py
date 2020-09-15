# -*- coding: utf-8 -*-
"""

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Extract data from the central part of galaxies to model with paintbox.

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
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from spectres import spectres
try:
    import ppxf_util as util
except:
    import ppxf.ppxf_util as util

import context

def collapse_cube(cubename, outfile, redo=False):
    """ Collapse a MUSE data cube to produce a white-light image and a
    noise image.

    The noise is estimated with the same definition of the DER_SNR algorithm.

    Input Parameters
    ----------------
    cubename : str
        Name of the MUSE data cube

    outfile : str
        Name of the output file

    redo : bool (optional)
        Redo calculations in case the outfile exists.
    """
    if os.path.exists(outfile) and not redo:
        return
    data = fits.getdata(cubename, 0)
    newdata = np.nanmean(data, axis=0)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.* data - \
           np.roll(data, 2, axis=0) - np.roll(data, -2, axis=0)), \
           axis=0)
    # noise = np.nanmean(np.sqrt(var[idx,:,:]), axis=0)
    hdu = fits.PrimaryHDU(newdata)
    hdu2 = fits.ImageHDU(noise)
    hdulist = fits.HDUList([hdu, hdu2])
    hdulist.writeto(outfile, overwrite=True)
    return

def find_center(img):
    data = fits.getdata(img)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + (3. * std)
    tbl = find_peaks(data, threshold, box_size=7)
    idx = np.argmax(tbl["peak_value"].data)
    tbl = tbl[idx]
    return tbl["x_peak"], tbl["y_peak"]

def extract_aperture_spectra(datacube, r=None, x0=44, y0=19):
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
    return specs

def calc_aperture(datacube, x0, y0, apertures=None, unit=None):
    apertures = np.arange(1, 8, 0.5) if apertures is None else apertures
    unit = u.arcsec if unit is None else unit
    snrs = np.zeros_like(apertures)
    for i, ap in enumerate(apertures):
        apspecs = extract_aperture_spectra(datacube, x0=x0, y0=y0,
                                           r=ap * unit)
        spec1d = np.nanmean(apspecs, axis=1)
        snrs[i] = der_snr(spec1d)
    idx = np.argmax(snrs)
    return apertures[idx] * unit

def der_snr(flux, axis=0):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro
    """
    signal = np.nanmedian(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.*flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)

    return signal / noise

def extract_max_sn_spectrum(datacube):
    dataimg = datacube.replace("cube", "cubeImg")
    collapse_cube(datacube, dataimg)
    x0, y0 = find_center(dataimg)
    aperture = calc_aperture(datacube, x0, y0)
    apspecs = extract_aperture_spectra(datacube, x0=x0, y0=y0,
                                       r=aperture)
    spec1d = np.nanmedian(apspecs, axis=1)
    spec1derr = np.nanstd(apspecs, axis=1) / np.sqrt(len(apspecs))
    return spec1d, spec1derr, aperture

def calc_dispersion(cube):
    hdr = fits.getheader(cube)
    wave = ((np.arange(hdr['NAXIS3']) + 1 - \
             hdr['CRPIX3']) * hdr['CDELT3'] + hdr['CRVAL3']) * u.m
    return wave

def data_reduction(galdir, scicube, tellcube, redo=False):
    sci_extracted = os.path.join(galdir, "spec1D.fits")
    tell_extracted = os.path.join(galdir, "tell1D.fits")
    if os.path.exists(sci_extracted) and not redo:
        return
    wave = calc_dispersion(scicube).to(u.micrometer).value
    wave_t = calc_dispersion(tellcube).to(u.micrometer).value
    spec1d, spec1derr, aper = extract_max_sn_spectrum(scicube)
    spec1d_t, spec1derr_t, aper_t = extract_max_sn_spectrum(tellcube)
    # Select wavelength to work and resampling
    w = np.where(np.isfinite(spec1d * spec1derr), wave, np.nan)
    idx = np.where((wave >= np.nanmin(w)) & (wave <= np.nanmax(w)))[0]
    wave = wave[idx]
    spec1d = spec1d[idx]
    spec1d_e = spec1derr[idx]
    spec1d[np.isnan(spec1d)] = 0
    spec1d_t[np.isnan(spec1d_t)] = 0
    spec1d_t = spectres(wave, wave_t, spec1d_t, fill=0.)
    # Saving results for science spectrum
    hdr = fits.getheader(scicube)
    hdr["APER"] = (aper.to(u.arcsec).value, "Aperture radius (arcsec)")
    hdu0 = fits.PrimaryHDU(header=hdr)
    mask = np.where(spec1d==0, 0, 1)
    table = Table([wave * u.micrometer, spec1d, spec1d_e, mask],
                  names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    hdu1 = fits.BinTableHDU(table)
    hdulist = fits.HDUList([hdu0, hdu1])
    hdulist.writeto(sci_extracted, overwrite=True)
    # Saving results of telluric cube
    hdr2 = fits.getheader(tellcube)
    hdr2["APER"] = (aper_t.to(u.arcsec).value, "Aperture radius (arcsec)")
    hdu0 = fits.PrimaryHDU(header=hdr2)
    mask_t = np.where(spec1d_t == 0, 0, 1)
    table = Table([wave * u.micrometer, spec1d_t, np.zeros_like(wave), mask_t],
                   names=["WAVE", "FLUX", "FLUX_ERR", "MASK"])
    hdu1 = fits.BinTableHDU(table)
    hdulist = fits.HDUList([hdu0, hdu1])
    hdulist.writeto(tell_extracted, overwrite=True)
    return

def run_molecfit(wdir, redo=False):
    cwd = os.getcwd()
    os.chdir(wdir)
    output = os.path.join(wdir, "spec1D_TAC.fits")
    if os.path.exists(output) and not redo:
        return
    scifile = os.path.join(wdir, "spec1D.fits")
    filelist = os.path.join(wdir, "filelist.txt")
    with open(filelist, "w") as f:
        f.write(scifile)
    tellfile = os.path.join(wdir, "tell1D.fits")
    stdheader = fits.getheader(tellfile)
    jd = float(stdheader["BARY_JD"])
    mjd = int(jd - 2400000.5)
    utc = datetime.strptime("12:19:32.5", "%H:%M:%S.%f")
    utc_seconds = utc.hour * 3600 + utc.minute * 60 + utc.second
    molecfit_params = {"user_workdir": os.getcwd(),
                       "filename": "tell1D.fits",
                       "listname": filelist,
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
    new_config_file = "wifis_zJ.par"
    with open(new_config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    subprocess.run(["bash", "/home/kadu/molecfit/bin/molecfit",
                    new_config_file])
    subprocess.run(["bash", "/home/kadu/molecfit/bin/calctrans",
                    new_config_file])
    # Changing columns to apply corrfilelist
    config["columns"] = "WAVE FLUX FLUX_ERR MASK"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    subprocess.run(["bash", "/home/kadu/molecfit/bin/corrfilelist",
                    config_file])
    shutil.copy(os.path.join(wdir, "molecfit", "spec1D_TAC.fits"), output)
    shutil.copy(os.path.join(wdir, "molecfit", "tell1D_TAC.fits"),
                os.path.join(wdir, "tell1D_TAC.fits"))
    os.chdir(cwd)
    return

def flux_calibration(galdir, redo=False):
    output = os.path.join(os.getcwd(), "scispec.fits")
    if os.path.exists(output) and not redo:
        return
    tellspec = os.path.join(galdir, "tell1D_TAC.fits")
    star = fits.getval(tellspec, "OBJECT").strip()
    # Make flux calibration
    two_mass = Vizier(columns=["*", "+_r"])
    std2mass = two_mass.query_object(star, catalog="II/246")[0][0]
    ref2mass = two_mass.query_object("Vega", catalog="II/246")[0][0]
    dmag = std2mass["Jmag"] - ref2mass["Jmag"]
    observed = Table.read(tellspec)
    idx = np.where(observed["tacflux"] > 0)[0]
    observed = observed[idx]
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
    table = Table.read(os.path.join(galdir, "spec1D_TAC.fits"))
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


if __name__ == "__main__":
    ps = 0.53884 * u.arcsec
    data_dir = os.path.join(context.data_dir, "WIFIS")
    home_dir = os.path.join(context.home, "center_imfs")
    if not os.path.exists(home_dir):
        os.mkdir(home_dir)
    galaxies = os.listdir(data_dir)
    apertures = np.array([5,])
    for galaxy in galaxies:
        print(galaxy)
        galdir = os.path.join(home_dir, galaxy)
        if not os.path.exists(galdir):
            os.mkdir(galdir)
        cube_dir = os.path.join(data_dir, galaxy)
        n = "2" if galaxy == "M85" else "1"
        scicube = os.path.join(cube_dir, "{}_combined_cube_{}.fits".format(
                                galaxy, n))
        tellcube = os.path.join(data_dir, "M85/HIP56736_combined_cube_1.fits")
        data_reduction(galdir, scicube, tellcube)
        run_molecfit(galdir)
        flux_calibration(galdir, redo=True)