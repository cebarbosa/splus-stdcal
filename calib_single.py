# -*- coding: utf-8 -*-
""" 

Created on 14/03/18

Author : Carlos Eduardo Barbosa

Use zero points and extinction coefficients determined with stdcal to calibrate
single exposure catalogs.

"""
from __future__ import print_function, division

import os
import sys
import yaml
import platform

import numpy as np
from astropy.table import Table
from astropy.io import fits
from multiprocessing import Pool
from tqdm import tqdm

import context
import misc

def calib_single_catalogs(zptable, flux_keys, survey, redo=False):
    """ Performs the calibration of catalogs from single exposure. """
    if platform.node() == "kadu-Inspiron-5557":
        data_dir = os.path.join(context.data_dir, "test_apply_photcal")
    else:
        data_dir = "/mnt/jype/MainSurvey/reduced"
    output_basedir = os.path.join(context.data_dir, "single")
    if not os.path.exists(output_basedir):
        os.mkdir(output_basedir)
    for i, sol in enumerate(zptable):
        print("Searching {} catalogs on date {} ({} / {})".format(survey,
              sol["date"], i+1, len(zptable)))
        wdir = os.path.join(data_dir, sol["date"])
        if not os.path.exists(wdir):
            continue
        output_dir = os.path.join(output_basedir, sol["date"])
        fzs = [_ for _ in os.listdir(wdir) if _.endswith(".fz") and
               _.startswith(survey)]
        for fz in tqdm(fzs):
            fzfile = os.path.join(wdir, fz)
            catfile = os.path.join(wdir, fz.replace(".fz", "files"),
                               fz.replace(".fz", ".apercorcat"))
            if not os.path.exists(fzfile):
                continue
            if not os.path.exists(catfile):
                continue
            ####################################################################
            # Getting metadata from header
            h = fits.getheader(fzfile, 1)
            X = h["AIRMASS"]
            exptime = h["EXPTIME"]
            band = h["FILTER"]
            obj = h["OBJECT"]
            output = os.path.join(output_dir, "{}_{}_{}.fits".format(
                fz.split("_")[0], obj.replace(" ", "").replace("_", "-"),
                band))
            if os.path.exists(output) and not redo:
                continue
            ###################################################################
            data = Table.read(catfile, 2, format="fits")
            for fluxkey in flux_keys:
                # Setting keywords
                ferrkey = "FLUXERR_APERCOR" if fluxkey == "FLUXAPERCOR" else \
                          fluxkey.replace("FLUX", "FLUXERR")
                magkey = 'MAGAPERCOR' if fluxkey == "FLUXAPERCOR" else \
                    fluxkey.replace("FLUX", "MAG")
                merrkey = "MAGERR_APERCOR" if magkey == "MAGAPERCOR" else \
                          magkey.replace("MAG", "MAGERR")
                # Get zeropoint and kappa
                zpkey = "zp_{}_{}".format(fluxkey, band)
                if zpkey not in sol.colnames:
                    data[fluxkey] = np.nan
                    data[ferrkey] = np.nan
                    data[magkey] = np.nan
                    data[merrkey] = np.nan
                    continue
                zp = sol[zpkey]
                zperr = sol["zperr_{}_{}".format(fluxkey, band)]
                kappa = sol["kappa_{}_{}".format(fluxkey, band)]
                kappaerr = sol["kappaerr_{}_{}".format(fluxkey, band)]
                if not np.all(np.isfinite([zp, zperr, kappa, kappaerr])):
                    data[fluxkey] = np.nan
                    data[ferrkey] = np.nan
                    data[magkey] = np.nan
                    data[merrkey] = np.nan
                    continue
                ################################################################
                # Adapt zeropoints according to observational parameters
                m0 = zp - kappa * X + 2.5 * np.log10(exptime)
                m0err = np.sqrt(np.power(zperr, 2) + np.power(kappaerr * X, 2))
                moff = 25.
                f0 = np.power(10, -0.4 * m0)
                f0err = np.abs(np.log(10) * 0.4 * f0 * m0err)
                ################################################################
                f = data[fluxkey]  # Instrumental values
                ferr = data[ferrkey] if ferrkey in data.keys() else \
                       np.zeros_like(f)
                data[fluxkey] = f * f0  # Applying calibration
                data[ferrkey] = np.sqrt(np.power(f * f0err, 2) +
                                       np.power(f0 * ferr, 2))
                ################################################################
                # Making the calibration in the magnitude fields
                m = data[magkey]  # Instrumental magnitudes
                merr = data[merrkey] if merrkey in data.colnames else \
                           np.zeros_like(m)
                data[magkey] = m + m0 - moff
                data[merrkey] = np.sqrt(m0err**2 + merr**2)
            # Saving data
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            data.write(output, overwrite=True)

def main():
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        default_file = "config_mode5.yaml"
        print("Using default config file ({})".format(default_file))
        config_files.append(default_file)
    for filename in config_files:
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # Set nights that will be calibrated
        nights = misc.select_nights(config)
        # Run sextractor to produce catalogs
        singles_dir = os.path.join(config["output_dir"], "single")
        if not os.path.exists(singles_dir):
            os.mkdir(singles_dir)
        pool = Pool(25)
        pool.map(f, nights)

if __name__ == "__main__":
    main()

    # flux_keys = ["FLUX_AUTO", "FLUX_ISO", "FLUXAPERCOR"]
    # survey = "STRIPE82"
    # redo=True
    # table = Table.read(os.path.join(context.tables_dir, "date_zps.fits"))
    # calib_single_catalogs(table, flux_keys, survey, redo=True)