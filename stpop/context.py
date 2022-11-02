# -*- coding: utf-8 -*-
"""

Created on 23/11/17

Author : Carlos Eduardo Barbosa

Project context.

"""

import os
import platform

import matplotlib
# matplotlib.use('Agg')
#matplotlib.use('qt5agg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


if platform.node() == "asmaclap04":
    home = "/Users/meyer/WIFIS/paintbox/wifis/stpop/data/"
elif platform.node() == 'wifis-monster':
    home = "/home/elliot/paintbox-wifis/stpop/data/"

data_dir = os.path.join(home, "data")
molecfit_exec_dir = os.path.join(home, "molecfit/bin")

fig_width = 3.54 # inches - A&A template

# Matplotlib settings
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
