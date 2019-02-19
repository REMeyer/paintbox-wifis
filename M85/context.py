# -*- coding: utf-8 -*-
""" 

Created on 09/10/18

Author : Carlos Eduardo Barbosa

Context file for the work on galaxy M85.

"""
import os
import getpass

import astropy.units as u
import matplotlib.pyplot as plt

if getpass.getuser() == "kadu":
    home = "/home/kadu/Dropbox/WIFIS/M85"
else:
    home = "/sto/home/cebarbosa/WIFIS/M85"

data_dir = os.path.join(home, "data")

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

# NED results
z = 0.002432
vsyst = 729 * u.km / u.s