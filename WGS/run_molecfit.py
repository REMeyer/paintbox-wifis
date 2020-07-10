# -*- coding: utf-8 -*-
"""
Created on 04/04/19
Author : Carlos Eduardo Barbosa
Runs molecfit on SPINS data.
"""
from __future__ import print_function, division

import os
import shutil
import yaml
import subprocess

def run_molecfit(params, outdir, redo=False):
    """ Runs molecfit. """
    if os.path.exists(outdir) and not redo:
        return
    cwd = os.getcwd()
    os.chdir(params["user_workdir"])
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "molecfit")
    # Copying default parameter files
    for fname in os.listdir(config_path):
        shutil.copy(os.path.join(config_path, fname), params["user_workdir"])
    config_file = os.path.join(params["user_workdir"], "wifis_zJ.par")
    with open(config_file) as f:
        config = yaml.load(f)
    for key in params.keys():
        if key in config.keys():
            config[key] = params[key]
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    molecfit = os.path.join(context.molecfit_exec_dir, "molecfit")
    calctrans = os.path.join(context.molecfit_exec_dir, "calctrans")
    corrfilelist = os.path.join(context.molecfit_exec_dir, "corrfilelist")
    subprocess.run(["bash", molecfit, config_file])
    subprocess.run(["bash", calctrans, config_file])
    # Changing columns to apply corrfilelist
    config["columns"] = "WAVE FLUX FLUX_ERR MASK"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    subprocess.run(["bash", corrfilelist, config_file])
    os.chdir(cwd)

def example():
    """ Example of how to run molecfit. """
    pass
