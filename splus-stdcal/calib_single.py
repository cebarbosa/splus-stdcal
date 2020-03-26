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

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm

import context

def load_date_zps(fname):
    """ Makes nested dictionary for calibration parameters.

    Input parameters
    ----------------
    fname: str
        Input filename FITS table.

    Output parameters
    -----------------
    dict
        Nested dictionary containing all zero points in the table. Results are
        accessed using dict[date][parname][filter], where parname includes
        zp, zperr, kappa and kappaerr, ntot (number of stars), nout (number
        of outliers).

    """
    table = Table.read(fname)
    results = {}
    for line in table:
        z, zerr, k, kerr, ntot, nout = {}, {}, {}, {}, {}, {}
        d = {}
        for band in context.bands:
            zpkey = "ZP_{}".format(band)
            if zpkey not in table.colnames:
                continue
            z[band] = line[zpkey]
            zerr[band] = line["eZP_{}".format(band)]
            k[band] = line["K_{}".format(band)]
            kerr[band] = line["eK_{}".format(band)]
            ntot[band] = line["Ntot_{}".format(band)]
            nout[band] = line["Ntot_{}".format(band)] - \
                         line["N_{}".format(band)]
        d["zp"] = z
        d["zperr"] = zerr
        d["kappa"] = k
        d["kappaerr"] = kerr
        d["ntot"] = ntot
        d["nout"] = nout
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
        Number of the HDU to be read.


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


def calib_no_outliers(nstars=3, noutliers=0):
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
                ntot = params["ntot"]
                nout = params["nout"]
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
                    # Select only nights without outliers
                    condition = (ntot[band] >= nstars) & \
                                (nout[band] <= noutliers)
                    if not condition:
                        continue
                    X = float(hdict["AIRMASS"][0])
                    exptime = float(hdict["EXPTIME"][0])
                    m0 = zp[band] - kappa[band] * X + 2.5 * np.log10(exptime)
                    m0err = np.sqrt(zperr[band]**2 + (kappaerr[band] * X)**2)
                    t["FILENAME"] = [single]
                    t["NIGHT"] = [date]
                    t["OBJECT"] = [hdict["OBJECT"][0]]
                    t["DATETIME"] = [hdict["DATE-OBS"][0]]
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