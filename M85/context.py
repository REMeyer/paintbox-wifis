# -*- coding: utf-8 -*-
"""

Created on 23/11/17

Author : Carlos Eduardo Barbosa

Project context.

"""

import os
import platform

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.config import config
from dustmaps import sfd

if platform.node() == "kadu-Inspiron-5557":
    home = "/home/kadu/Dropbox/WGS/M85"
elif platform.node() in ["uv100", "alphacrucis"]:
    home = "/sto/home/cebarbosa/WGS/M85"

data_dir = os.path.join(home, "data")
molecfit_exec_dir = os.path.join(home, "molecfit/bin")