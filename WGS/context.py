# -*- coding: utf-8 -*-
""" 

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Context file for the work on galaxy M85.

"""
import os

import astropy.units as u
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, "Dropbox/WGS")
molecfit_exec_dir = os.path.join(home_dir, "molecfit/bin")

data_dir = os.path.join(project_dir, "data")

plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

FWHM = 2.53 # WIFIS resolution in Angstrom