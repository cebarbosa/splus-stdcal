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

import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
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

def get_standard_data(nights, data_dir, outdir_root, redo=False):
    """ Search for catalogs and get catalog information for standard stars. """
    catalog = "I/345/gaia2"
    for night in nights:
        outdir = os.path.join(outdir_root, night)
        if os.path.exists(outdir) and not redo:
            continue
        path = os.path.join(data_dir, night)
        images = [_ for _ in os.listdir(path) if _.endswith("_proc.fits")]
        images = sorted([_ for _ in images if _.startswith("EXTMONI")])
        cats = [os.path.join(path, _.replace(".fits", "files"),
                             _.replace(".fits", ".apercorcat")) for _ in images]
        for img, cat in zip(images, cats):
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            if not os.path.exists(cat):
                continue
            outcat = os.path.join(outdir, "{}_cat.fits".format(
                img.split("_proc")[0]))
            if os.path.exists(outcat) and not redo:
                continue
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
            input(404)

def main():
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        print("Configuration file not found. Using default configurations.")
        config_files.append("config_oldgain.yaml")
    for filename in config_files:
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if not os.path.exists(config["output_dir"]):
            os.mkdir(config["output_dir"])
        # Set nights that will be calibrated
        nights = select_nights(config)
        extmoni_dir = os.path.join(config["output_dir"], "extmoni")
        if not os.path.exists(extmoni_dir):
            os.mkdir(extmoni_dir)
        get_standard_data(nights, config["singles_dir"], extmoni_dir)
    print("Done!")

if __name__ == "__main__":
    main()

