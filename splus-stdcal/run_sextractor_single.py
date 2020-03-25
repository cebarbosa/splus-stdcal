# -*- coding: utf-8 -*-
""" 

Created on 13/01/20

Author : Carlos Eduardo Barbosa

Running SExtractor into single exposure images.

"""
from __future__ import print_function, division

import os
import sys
import yaml
from subprocess import call
from functools import partial

from multiprocessing import Pool

import misc

def run_sextractor_single(data_dir, outdir, night, redo=False):
    """ Runs SExtractor on stamps of standard stars. """
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                              "config")
    sex_config_file = os.path.join(config_dir, "stdcal.sex")
    sex_params_file = os.path.join(config_dir, "stdcal.param")
    filter_file = os.path.join(config_dir, "gauss_3.0_5x5.conv")
    starnnw_file = os.path.join(config_dir, "default.nnw")
    wdir = os.path.join(data_dir, night)
    if not os.path.exists(wdir):
        return
    images = sorted([_ for _ in os.listdir(wdir) if
                     _.endswith("_proc.fits") and
                     not _.startswith("EXTMONI")])
    for img in images:
        imgfile = os.path.join(wdir, img)
        cat_dir = os.path.join(outdir, night)
        if not os.path.exists(cat_dir):
            os.mkdir(cat_dir)
        sexcat = os.path.join(cat_dir, img.replace(".fits", ".cat"))
        if os.path.exists(sexcat) and not redo:
            continue
        tmp_file = os.path.join(cat_dir, img)
        call(["funpack", "-O", tmp_file, imgfile])
        call(["sextractor", tmp_file, "-c", sex_config_file,
              "-PARAMETERS_NAME", sex_params_file,
              "-FILTER_NAME", filter_file, "-STARNNW_NAME", starnnw_file,
              "-CATALOG_NAME", sexcat])
        os.remove(tmp_file)


def main():
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        default_file = "config_mode5.yaml"
        print("Using default config file ({})".format(default_file))
        config_files.append(default_file)
    for filename in config_files:
        with open(filename) as f:
            config = yaml.load(f)
        # Set nights that will be calibrated
        nights = misc.select_nights(config)
        # Run sextractor to produce catalogs
        singles_dir = os.path.join(config["output_dir"], "single")
        if not os.path.exists(singles_dir):
            os.mkdir(singles_dir)
        f = partial(run_sextractor_single, config["singles_dir"], singles_dir)
        pool = Pool(8)
        pool.map(f, nights)

if __name__ == "__main__":
    main()