# -*- coding: utf-8 -*-
"""

Originally created on 23/11/17

Author : Carlos Eduardo Barbosa
Updated by: R Elliot Meyer (2022/2023)

Project context.

"""

import os
import platform

import matplotlib
import numpy as np
# matplotlib.use('Agg')
#matplotlib.use('qt5agg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#### Run Settings
# Observed target settings
sample = ["M85", "NGC5557"] # Target names
fit_regions = ["R1", "R2"] # Fitting/spectral regions
# Dictionary of observing dates, for filepath
obsdates = {"M85": {'R1': "20210324", 'R2': '20210324'}, 
        "NGC5557": {'R1':"20200709", "R2":"20210324"}}
# Kinematic Priors. Format is (sigma, width). If not using, set to None
kin_priors = {"M85": {'R1': (142,12), 'R2': (177,17)}, 
        "NGC5557": {'R1': (220,30), "R2": (164,25)}}
V = {"M85": 729, "NGC5557": 3219} # Estimate of 
kinematic_fit = True
masking_regions = []

dirsuffix = '20230220_KinematicFitTest'
#forcedir = 'FullSpectralFitR1'
forcedir = None

#### Model Settings
postprocessing = True# if getpass.getuser() == "kadu" else False
ssp_model = "CvD"
loglike = "studt"
nsteps = 4000
porder = 45

labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]", "Age": "Age (Gyr)",
              "Na": "[Na/Fe]" if ssp_model == "emiles" else "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]",
              "V": "$V_*$ (km/s)", "sigma": "$\sigma_*$ (km/s)",
              "alpha_Ks": r"$\alpha_{\rm Ks}$",
              "M2L_Ks": r"(M/L)$_{\rm Ks}$"}
#elements = ["Na", "Fe", "Ca", "K"]
#elements = ["Na", "Fe", "K"]
elements = None

# Absorption Line definitions
#WIFIS Defs
bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 
                12240, 11905]
#bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 
#                12260, 11935]
#linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 
#                12309, 11935]
#linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 
#                12333, 11965]
#redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 
#                12360, 12005]
redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 
                12390, 12025]
line_name = np.array(['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB',\
        'NaI127', 'NaI123','CaII119'])

#### Backend Settings
if platform.node() == "asmaclap04":
    home = "/Users/meyer/WIFIS/paintbox/wifis/stpop/data/"
elif platform.node() == 'wifis-monster':
    home = "/home/elliot/paintbox-wifis/stpop/data/"
data_dir = os.path.join(home, "data")
molecfit_exec_dir = os.path.join(home, "molecfit/bin")

#### Matplotlib Settings
fig_width = 3.54 # inches - A&A template

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

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set tick width
width = 0.5
majsize = 4
minsize = 2
plt.rcParams['xtick.major.size'] = majsize
plt.rcParams['xtick.major.width'] = width
plt.rcParams['xtick.minor.size'] = minsize
plt.rcParams['xtick.minor.width'] = width
plt.rcParams['ytick.major.size'] = majsize
plt.rcParams['ytick.major.width'] = width
plt.rcParams['ytick.minor.size'] = minsize
plt.rcParams['ytick.minor.width'] = width
plt.rcParams['axes.linewidth'] = width