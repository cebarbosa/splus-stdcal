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
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm

import context

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

def load_date_zps(fname, nstars=3, noutliers=0):
    """ Makes nested dictionary for calibration parameters.

    Input parameters
    ----------------
    fname: str
        Input filename FITS table.

    nstars: int (optional)
        Minimum number of stars used in the calibration of the night.

    noutliers: int (optional)
        Maximum number of outliers in the fit.

    Output parameters
    -----------------
    dict
        Nested dictionary containing all zero points in the table. Results are
        accessed using dict[date][parname][filter], where parname includes
        zp, zperr, kappa and kappaerr.

    """
    table = Table.read(fname)
    results = {}
    for line in table:
        z, zerr, k, kerr = {}, {}, {}, {}
        d = {}
        for band in context.bands:
            zpkey = "ZP_{}".format(band)
            if zpkey not in table.colnames:
                continue
            # Select only nights without outliers
            ntot = line["Ntot_{}".format(band)]
            n = line["N_{}".format(band)]
            nout = ntot - n
            condition = (ntot >= nstars) & (nout <= noutliers)
            if not condition:
                continue
            z[band] = line[zpkey]
            zerr[band] = line["eZP_{}".format(band)]
            k[band] = line["K_{}".format(band)]
            kerr[band] = line["eK_{}".format(band)]
        d["zp"] = z
        d["zperr"] = zerr
        d["kappa"] = k
        d["kappaerr"] = kerr
        results[line["DATE"]] = d
    return results

def load_header_fits_rec(catfile, hdu=1):
    """ Load header parameters from a LDAC_FITS file (not compatible with
    astropy) into a dictionary.

    Input parameters
    ----------------
    catfile: str
        Path for the LDAC_FITS file

    hdu: int (optional)
        Number of the hdu to be read.


    Output parameters
    -----------------
    dict
        Dictionary containing the header information, including values and
        comments (Warning: always strings).

    """
    hdata = str(fits.getdata(catfile, hdu=hdu)[0])[12:-23]
    hdata = hdata.replace(",", " ").replace("'", "").replace(
        '"', "")
    hdata = [_.strip() for _ in hdata.split("\n")]
    hdict = {}
    for line in hdata:
        if len(line.split("=")) < 2:
            continue
        keywd, val = line.split("=", 1)
        hdict[keywd.strip()] = [_.strip() for _ in val.split("/")]
    return hdict


def calib_no_outliers():
    """ Calibration script for the cases without outliers. """
    config_files = ["config_mode0.yaml", "config_mode5.yaml"]
    for j, config_file in enumerate(config_files):
        print("Config file {} / {}".format(j+1, len(config_files)))
        with open(config_file, "r") as f:
            config = yaml.load(f)
        calib_dir = os.path.join(config["main_dir"], "calib")
        outdir = os.path.join(config["main_dir"], "zeropoints")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        single_dir_out = os.path.join(outdir, "single")
        if not os.path.exists(single_dir_out):
            os.mkdir(single_dir_out)
        calibs = [_ for _ in os.listdir(calib_dir) if _.startswith(
                          config["name"])]
        latest = sorted(calibs)[-1]
        single_dir = os.path.join(config["main_dir"], "single")
        for phot in config["sex_phot"]:
            print("Processing photometry mode {}".format(phot))
            fname = os.path.join(calib_dir, latest, phot, "zps-dates.fits")
            zpdata = load_date_zps(fname)
            output = os.path.join(single_dir_out, "{}-{}.fits".format(
                                  config["name"], phot))
            outtable = []
            for i, (date, params) in enumerate(tqdm(zpdata.items())):
                zp = params["zp"]
                zperr = params["zperr"]
                kappa = params["kappa"]
                kappaerr = params["kappaerr"]
                data_dir = os.path.join(single_dir, date)
                if not os.path.exists(data_dir):
                    continue
                singles = sorted(os.listdir(data_dir))
                for single in singles:
                    t = Table()
                    catfile = os.path.join(data_dir, single)
                    hdict = load_header_fits_rec(catfile)
                    band = hdict["FILTER"][0]
                    if band not in zp:
                        continue
                    X = float(hdict["AIRMASS"][0])
                    exptime = float(hdict["EXPTIME"][0])
                    m0 = zp[band] - kappa[band] * X + 2.5 * np.log10(exptime)
                    m0err = np.sqrt(zperr[band]**2 + (kappaerr[band] * X)**2)
                    t["OBJECT"] = [hdict["OBJECT"][0]]
                    t["DATE"] = [hdict["DATE-OBS"][0]]
                    t["FILTER"] = [band]
                    t["EXPTIME"] = [exptime]
                    t["GAIN"] = [float(hdict["GAIN"][0])]
                    t["ZP"] = [np.round(m0, 5)]
                    t["ZPERR"] = [np.round(m0err, 5)]
                    outtable.append(t)
            outtable = vstack(outtable)
            outtable.write(output, format="fits", overwrite=True)

if __name__ == "__main__":
    calib_no_outliers()