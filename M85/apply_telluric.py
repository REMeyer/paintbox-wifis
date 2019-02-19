# -*- coding: utf-8 -*-
""" 

Created on 22/10/18

Author : Carlos Eduardo Barbosa

Apply telluric correction to combined spectra

"""

from __future__ import print_function, division

import os

from astropy.table import Table
import matplotlib.pyplot as plt

import context

def apply_molecfit(targetSN, molecfit_file):
    """ Applying calculated transmission curve to spectra. """
    trans = Table.read(molecfit_file)
    data_dir = os.path.join(context.data_dir, "spec1d_sn{}".format(targetSN))
    outdir = os.path.join(context.data_dir, "molecfited_sn{}".format(
        targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    specs = sorted(os.listdir(data_dir))
    for spec in specs:
        data = Table.read(os.path.join(data_dir, spec))
        plt.plot(data["WAVE"], data["FLUX"], "-")
        data["FLUX"] /= trans["mtrans"]
        data["FLUX_ERR"] /= trans["mtrans"]
        plt.plot(data["WAVE"], data["FLUX"], "-")
        plt.show()
        data.write(os.path.join(outdir, spec), overwrite=True)


if __name__ == "__main__":
    targetSN = 40
    molecfit_file = os.path.join(context.home,
                    "data/molecfit/output/HIP56736_spec1D_TAC.fits")
    apply_molecfit(targetSN, molecfit_file)