""" Calculation of the M/L ratio for Elliot's traces. """
import os

import numpy as np
from astropy.table import Table
import paintbox as pb

def table2array(t):
    return np.lib.recfunctions.structured_to_unstructured(t.as_array())


def loadm2l():
    global home_dir
    m2l_table = Table.read(os.path.join(home_dir,
                                        "tables/FSPS_magnitudes.fits"))
    m2l_table = m2l_table[m2l_table["age"] >= 1]
    m2l_table = m2l_table[m2l_table["age"] < 16]
    inpars = ["logzsol", "age", "imf1", "imf2"]
    intable = m2l_table[inpars]
    outpars = ["2mass_ks"]
    data = table2array(m2l_table[outpars])
    wave = np.array([21590])
    m2l = pb.ParametricModel(wave, m2l_table[inpars], data)
    return m2l

def print_summary(a, name):
    m = np.percentile(a, 50)
    uerr = np.percentile(a, 84) - m
    lerr = m - np.percentile(a, 16)
    print(f"{name}: {m:.2f} +{uerr:.2f}-{lerr:.2f}")
    return


if __name__ == "__main__":
    home_dir = "/home/kadu/Dropbox/SPINS"
    m2l = loadm2l()
    traces_dir = os.path.join(home_dir, "traces")
    trace_files = [_ for _ in os.listdir(traces_dir) if _.endswith("mcmc")]
    for filename in trace_files:
        print(filename)
        t = Table.read(os.path.join(traces_dir, filename), format="ascii")
        data = table2array(t[["Z", "Age", "x1", "x2"]])
        data_ml = m2l(data)
        data[:,2] = 1.3
        data[:,3] = 2.3
        kroupa_ml = m2l(data)
        alpha = data_ml / kroupa_ml
        print_summary(data_ml, "M/L")
        print_summary(alpha, "alpha")


