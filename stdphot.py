# -*- coding: utf-8 -*-
""" 

Created on 07/08/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import sys
import yaml
from subprocess import call
from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from tqdm import tqdm

import sip_tpv

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

def make_std_cutout(nights, data_dir, outdir_root, redo=False,
                      skip_existing_nights=True, cutout_size=None):
    """ Search for catalogs and get catalog information for standard stars. """
    print("Initial number of nights: {}".format(len(nights)))
    stdcoords = Table.read("tables/stdcoords.fits", format="fits")
    header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                   "INSTRUME", "AIRMASS"]
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
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outcat = os.path.join(outdir, "{}_cat.fits".format(
                img.split("_proc")[0]))
            if os.path.exists(outcat) and not redo:
                continue
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            # Find coordinates of standard star using Vizier
            h = fits.getheader(os.path.join(path, img))
            sip_tpv.pv_to_sip(h)
            wcs = WCS(h, relax=True)
            obj = h["object"].replace(" ", "").lower()
            coords = stdcoords[np.where(obj==stdcoords["STAR"])[0]]
            starcoords = SkyCoord(ra=coords["RA_ICRS"], dec=coords["DE_ICRS"],
                                   distance=coords["distance"],
                                  pm_ra_cosdec=coords['pmRA'],
                                  pm_dec=coords['pmDE'],
                                  obstime=Time(2015.5, format='decimalyear'))

            c = starcoords.apply_space_motion(new_obstime=Time(night))
            x0, y0 = wcs.all_world2pix(c.ra, c.dec, 1)
            # Make cutout
            imgout = os.path.join(outdir, "{}_stamp.fits".format(
                                  img.split("_proc")[0]))
            data = fits.getdata(os.path.join(path, img))
            try:
                cutout = Cutout2D(data, position=(x0, y0),
                                  size=cutout_size * u.pixel, wcs=wcs)
            except:
                continue
            hdu = fits.ImageHDU(cutout.data, header=cutout.wcs.to_header())
            for key in header_keys:
                hdu.header[key] = h[key]
            hdu.header["IMAGE"] = img.split("_proc")[0]
            hdu.header["DATE"] = night
            hdu.writeto(imgout, overwrite=True)

def run_sextractor(data_dir, nights, redo=False):
    """ Runs SExtractor on stamps of standard stars. """
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                              "config")
    sex_config_file = os.path.join(config_dir, "stdcal.sex")
    sex_params_file = os.path.join(config_dir, "stdcal.param")
    filter_file = os.path.join(config_dir, "gauss_3.0_5x5.conv")
    starnnw_file = os.path.join(config_dir, "default.nnw")
    print("Running SExtractor...")
    for i, night in enumerate(tqdm(nights)):
        wdir = os.path.join(data_dir, night)
        stamps = [_ for _ in os.listdir(wdir) if _.endswith("stamp.fits")]
        for stamp in stamps:
            imgfile = os.path.join(data_dir, night, stamp)
            sexcat = imgfile.replace(".fits", ".cat")
            if os.path.exists(sexcat) and not redo:
                continue
            call(["sextractor", imgfile, "-c", sex_config_file,
                  "-PARAMETERS_NAME", sex_params_file,
                  "-FILTER_NAME", filter_file, "-STARNNW_NAME", starnnw_file,
                  "-CATALOG_NAME", sexcat])

def join_tables(data_dir, nights, output, rtol=5, redo=True):
    """ Join standard star catalogs into one table. """
    header_keys = ["DATE", "IMAGE", "OBJECT", "FILTER", "EXPTIME", "GAIN",
                   "AIRMASS"]
    print("Loading table data...")
    outtable = []
    for night in tqdm(nights):
        wdir = os.path.join(data_dir, night)
        stamps = sorted([_ for _ in os.listdir(wdir) if
                         _.endswith("stamp.fits")])
        for stamp in stamps:
            stampfile = os.path.join(wdir, stamp)
            catfile = stampfile.replace(".fits", ".cat")
            if not os.path.exists(os.path.join(wdir, catfile)):
                continue
            h = fits.getheader(stampfile, ext=1)
            x0, y0 = 0.5 * h["NAXIS1"], 0.5 * h["NAXIS2"]
            cat = Table.read(catfile, hdu=2)
            if len(cat) == 0:
                continue
            sep = np.sqrt((cat["X_IMAGE"] - x0)**2 + (cat["Y_IMAGE"] - y0)**2)
            idx = np.argmin(sep)
            cat = cat[idx]
            if sep[idx] > rtol:
                continue
            meta = Table([[h[key]] for key in header_keys], names=header_keys)
            t = hstack([meta, cat])
            outtable.append(t)
    print("Joining tables and saving the results...")
    outtable = vstack(outtable)
    outtable.rename_column("OBJECT", "STAR")
    outtable["STAR"] = [_.lower().replace(" ", "") for _ in outtable["STAR"]]
    outtable.write(output, format="fits", overwrite=True)
    return

def main():
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        default_file = "config_mode5.yaml"
        print("Using default config file ({})".format(default_file))
        config_files.append(default_file)
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
        nights = sorted(nights)
        extmoni_dir = os.path.join(config["output_dir"], "extmoni")
        if not os.path.exists(extmoni_dir):
            os.mkdir(extmoni_dir)
        make_std_cutout(nights, config["singles_dir"], extmoni_dir,
                          cutout_size=config["cutout_size"])
        run_sextractor(extmoni_dir, nights, redo=config["sex_redo"])

        outtable = os.path.join(config["output_dir"],
                                "phottable_{}.fits".format(config["name"]))
        join_tables(extmoni_dir, nights, outtable, rtol=config["rtol"])
    print("Done!")

if __name__ == "__main__":
    main()