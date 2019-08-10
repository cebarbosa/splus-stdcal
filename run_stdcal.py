# -*- coding: utf-8 -*-
""" 

Created on 07/08/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import sys
import yaml
from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import sewpy
from tqdm import tqdm

def select_nights(config):
    """ Select all nights in the configuration file range. """
    first_night = datetime.strptime(str(config["first_night"]), '%Y-%m-%d')
    last_night = datetime.strptime(str(config["last_night"]), '%Y-%m-%d')
    all_nights = sorted([_ for _ in sorted(os.listdir(config["singles_dir"])) \
                if os.path.isdir(os.path.join(config["singles_dir"], _))])
    nights = []
    for direc in all_nights:
        try:
            night = datetime.strptime(direc, "%Y-%m-%d")
        except:
            continue
        if first_night <= night <= last_night:
            nights.append(direc)
    return nights

def get_standard_data(nights, data_dir, outdir_root, redo=False,
                      skip_existing_nights=True, cutout_size=None):
    """ Search for catalogs and get catalog information for standard stars. """
    print("Initial number of nights: {}".format(len(nights)))
    catalog = "I/345/gaia2"
    header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                   "INSTRUME"]
    # Skip all processing related to nights already processed before
    # That makes the processing much faster if we just need to update the
    # analysis for new observing nights
    if skip_existing_nights:
        nights = [_ for _ in nights if not os.path.exists(os.path.join(
                  outdir_root, _))]
    print("Total nights to be processed in this run: {}".format(len(nights)))
    for i, night in enumerate(nights):
        print("Processing EXTMONI images of date {} ({}/{})".format(night,
              i+1, len(nights)))
        outdir = os.path.join(outdir_root, night)
        path = os.path.join(data_dir, night)
        images = [_ for _ in os.listdir(path) if _.endswith("_proc.fits")]
        images = sorted([_ for _ in images if _.startswith("EXTMONI")])
        if len(images) == 0:
            print("No processed images for this night. Continuing...")
            continue
        for img in tqdm(images):
            cat = os.path.join(path, img.replace(".fits", "files"),
                             img.replace(".fits", ".apercorcat"))
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            if not os.path.exists(cat):
                continue
            outcat = os.path.join(outdir, "{}_cat.fits".format(
                img.split("_proc")[0]))
            if os.path.exists(outcat) and not redo:
                continue
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            # Find coordinates of standard star using Vizier
            h = fits.getheader(os.path.join(path, img))
            obj = h["object"]
            q = Vizier.query_object(obj, catalog=catalog)[catalog]
            starcoords = SkyCoord(q["RA_ICRS"].mean(), q["DE_ICRS"].mean(),
                                  unit=(u.degree, u.degree))
            # Find line in the catalog containing the information of star
            catdata = Table.read(cat, hdu=2)
            catcoords = SkyCoord(catdata["ALPHA_J2000"],
                                 catdata["DELTA_J2000"],
                                 unit=(u.degree, u.degree))
            idx, d2d, d3d = starcoords.match_to_catalog_sky(catcoords)
            stardata = catdata[idx]
            # Add information from the header to the catalog line
            t = Table()
            t["IMAGE"] = [img.split("_proc")[0]]
            t["DATE"] = [night]
            t["STAR"] = [obj]
            t["FILTER"] = h["filter"]
            t["AIRMASS"] = h["airmass"]
            t["EXPTIME"] = h["exptime"]
            t["D2D"] = [d2d.to("arcsec")]
            cat = hstack([stardata, t])
            cat.write(outcat, format="fits", overwrite=True)
            # Make cutout
            imgout = os.path.join(outdir, "{}_stamp.fits".format(
                                  img.split("_proc")[0]))
            data = fits.getdata(os.path.join(path, img))
            cutout = Cutout2D(data, position=(cat["X_IMAGE"], cat["Y_IMAGE"]),
                              size=cutout_size * u.pixel)
            hdu = fits.ImageHDU(cutout.data)
            for key in header_keys:
                hdu.header[key] = h[key]
            hdu.writeto(imgout, overwrite=True)

def run_sextractor(data_dir, nights, coords):
    """ Runs SExtractor on stamps of standard stars. """
    config_dir = os.path.split(os.path.realpath(__file__))[0]
    sex_config_file = os.path.join(config_dir, "stdcal.sex")
    sex_params_file = os.path.join(config_dir, "stdcal.param")
    with open(sex_params_file) as f:
        sexpars = [_.strip() for _ in f.readlines()]
    for i, night in enumerate(nights):
        print("Running SExtractor on night {} ({}/{})".format(night, i+1,
                                                              len(nights)))
        wdir = os.path.join(data_dir, night)
        stamps = [_ for _ in os.listdir(wdir) if _.endswith("stamp.fits")]
        sew = sewpy.SEW(sexpath="sextractor", configfilepath=sex_config_file,
                        workdir=wdir, params=sexpars, loglevel="CRITICAL",
                        nice=19)
        for stamp in tqdm(stamps):
            imgfile = os.path.join(data_dir, night, stamp)
            sexcat = imgfile.replace("stamp", "sexcat")
            cat = sew(imgfile)["table"]
            rsep = np.sqrt((cat["X_IMAGE"] - coords[0])**2 +
                                    (cat["Y_IMAGE"] - coords[1])**2)
            idx = np.argmin(rsep)
            if rsep[idx] <= 2:
                cat = Table(cat[idx])
                cat.write(sexcat, overwrite=True)

def main():
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        print("Configuration file not found. Using default configurations.")
        config_files.append("config_mode5.yaml")
    for filename in config_files:
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if not os.path.exists(config["output_dir"]):
            os.mkdir(config["output_dir"])
        # Set nights that will be calibrated
        nights = select_nights(config)
        # Removing nights without standard stars
        nights = [night for night in nights if
                  any(s.startswith("EXTMONI") and s.endswith("_proc.fits")
           for s in os.listdir(os.path.join(config["singles_dir"], night)))]
        extmoni_dir = os.path.join(config["output_dir"], "extmoni")
        if not os.path.exists(extmoni_dir):
            os.mkdir(extmoni_dir)
        get_standard_data(nights, config["singles_dir"], extmoni_dir,
                          cutout_size=config["cutout_size"])
        starcoords = np.ones(2) * config["cutout_size"] * 0.5
        run_sextractor(extmoni_dir, nights, starcoords)

    print("Done!")

if __name__ == "__main__":
    main()

