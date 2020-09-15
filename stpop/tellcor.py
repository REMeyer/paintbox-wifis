"""
Determines the telluric correction using the method described by
Vacca et al 2003.
"""

import os

import numpy as np
from astropy.table import Table
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

import context
import paintbox as pb

if __name__ == "__main__":
    obs_file = os.path.join(context.home, "center_imfs/M85/tell1D.fits")
    observed = Table.read(obs_file)
    template = Table.read(os.path.join(context.data_dir,
                                       "rieke2008/table7.fits"))
    template.rename_column("lambda", "WAVE")
    template.rename_column("Vega", "FLUX")
    # Cropping template in wavelength
    dlam = 0.1
    idx = np.where((template["WAVE"] >= observed["WAVE"].min() - dlam) &
                    (template["WAVE"] <= observed["WAVE"].max() + dlam) )
    template = template[idx]


