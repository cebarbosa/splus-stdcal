# -*- coding: utf-8 -*-
""" 

Created on 01/12/17

Author : Carlos Eduardo Barbosa

Determination of solution for photometric calibration.

"""
from __future__ import print_function, division

import os
import sys
import yaml

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, vstack, join, hstack
import speclite.filters
import pymc3 as pm

import context

def load_java_system(version):
    """ Use speclite to load Javalambre's SPLUS filters for convolution. """
    tables_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                               "tables")
    filters_dir = os.path.join(tables_dir, "filter_curves", version)
    ftable = ascii.read(os.path.join(tables_dir, "filter_lam_filename.txt"))
    filternames = []
    for f in ftable:
        filtname = f["filter"]
        fname = os.path.join(filters_dir, f["filename"])
        fdata = np.loadtxt(fname)
        fdata = np.vstack(((fdata[0,0]-1, 0), fdata, (fdata[-1,0]+1, 0)))
        w = fdata[:,0] * u.AA
        response = np.clip(fdata[:,1], 0., 1.)
        speclite.filters.FilterResponse(wavelength=w,
                  response=response, meta=dict(group_name="java",
                  band_name=filtname.upper()))
        filternames.append("java-{}".format(filtname))
    java = speclite.filters.load_filters(*filternames)
    return java

def get_splus_magnitudes(stars, library, filter_curves):
    """ Obtain magnitude of stars used in the calibration in the
    S-PLUS system. """
    stds_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                               "tables/stdstars", library)
    java = load_java_system(filter_curves)
    mags = {}
    for star in stars:
        specfile = os.path.join(stds_dir, "f{}.dat".format(star))
        if not os.path.exists(specfile):
            continue
        wave, flux = np.loadtxt(specfile, usecols=(0,1), unpack=True)
        wave = wave * u.AA
        flux = flux * (10 ** -16) * u.erg / u.cm / u.cm / u.s / u.AA
        m = java.get_ab_magnitudes(flux, wave, mask_invalid=True)
        mags[star] = m
    return mags

def single_band_calib(data, outdb, redo=False):
    """ Determining zero points for single band data.
    """
    if os.path.exists(outdb) and not redo:
        return
    data = data.to_pandas()
    nights = list(data.DATE.unique())
    nights_lookup = dict(zip(nights, range(len(nights))))
    data["night"] = data.DATE.copy()
    data['night'] = data.night.replace(nights_lookup).values
    N = len(nights)
    with pm.Model():
        # Hyperpriors
        Mzp = pm.Normal('Mzp', mu=20., sd=2)
        Szp = pm.HalfCauchy('Szp', 5)
        Mkappa = pm.Normal('Mkappa', mu=0., sd=1)
        Skappa = pm.HalfCauchy(r'Skappa', 5)
        # Randon intercepts
        zp = pm.Cauchy("zp", alpha=Mzp, beta=Szp, shape=N)
        # Common slope
        kappa = pm.Cauchy(r"kappa", alpha=Mkappa, beta=Skappa, shape=N)
        # Model error
        eps = pm.HalfCauchy(r"eps", 5)
        # Expected value
        linear_regress = zp[data["night"]] - kappa[data["night"]] * data[
            "AIRMASS"]
        pm.Cauchy('y', alpha=linear_regress, beta=eps, observed=data["DELTAMAG"])
        trace = pm.sample()
    pm.save_trace(trace, outdb, overwrite=True)
    summary = []
    for night, idx in nights_lookup.items():
        t = Table([[night], [np.mean(trace["zp"][:, idx])],
                   [np.std(trace["zp"][:, idx])],
                   [np.mean(trace["kappa"][:, idx])],
                   [np.std(trace["kappa"][:, idx])]],
                  names=["date", "zp", "zperr", "kappa", "kappaerr"])
        summary.append(t)
    summary = vstack(summary)
    summary.write("{}.txt".format(outdb), overwrite=True, format="ascii")
    return

def merge_zp_tables(redo=False):
    """ Make single table containing zero points for all types of photometric
    methods. """
    print("Producing table with all zero points...")
    output = os.path.join(context.tables_dir, "date_zps.fits")
    flux_keys = ["FLUX_AUTO", "FLUX_ISO", "FLUXAPERCOR"]
    wdir = os.path.join(context.home_dir, "zps")
    epochs = [_ for _ in os.listdir(wdir) if os.path.isdir(os.path.join(wdir,
                                                                      _ ))]
    etables = []
    for epoch in epochs:
        zptables = None
        for fkey in flux_keys:
            for band in context.bands:
                tablename = os.path.join(wdir, epoch, "zp_{}_{}.fits".format(
                    fkey, band))
                if not os.path.exists(tablename):
                    continue
                zptable = Table.read(tablename)
                for s in ["zp", "zperr", "kappa", "kappaerr"]:
                    incol = "{}_{}".format(band, s)
                    zptable.rename_column(incol, "{}_{}_{}".format(s, fkey,
                                                                   band))
                if zptables is None:
                    zptables = zptable
                else:
                    zptables = join(zptables, zptable, join_type="outer")
        etables.append(zptables)
    etables = vstack(etables)
    etables.write(output, overwrite=True)
    print("Do not forget to upload the current version of the zeropoints!")
    return etables

def main():
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        print("Configuration file not found. Using default configurations.")
        config_files.append("config_mode5.yaml")
    for filename in config_files:
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        phot_file = os.path.join(config["output_dir"],
                                "phottable_{}.fits".format(config["name"]))
        phot = Table.read(phot_file)
        ########################################################################
        # Remove flagged stars
        flagged_stars = [_.lower().replace(" ", "") for _ in config[
            "flagged_stars"]]
        badidx = []
        for fstar in flagged_stars:
            idx = np.where(phot["STAR"]==fstar)[0]
            badidx.append(idx)
        badidx = np.hstack(badidx)
        goodidx = np.arange(len(phot))
        goodidx = np.delete(goodidx, badidx)
        phot = phot[goodidx]
        ########################################################################
        # Add columns to the photometry table containing the model magnitudes
        stars = set(phot["STAR"])
        model_mags = get_splus_magnitudes(stars, config["stdlib"],
                                          config["filters_version"])
        modelmag = np.zeros(len(phot)) * np.nan
        for star in stars:
            if star not in model_mags:
                print("Star not found in database: {}".format(star))
                continue
            for band in context.bands:
                fname = "java-{}".format(band)
                idx = np.where((phot["STAR"]==star) & (phot["FILTER"]==band))[0]
                modelmag[idx] = float(model_mags[star][fname])
        phot["MODELMAG"] = modelmag
        phot["OBSMAG"] = -2.5 * np.log10(phot["FLUX_APER"] / phot["EXPTIME"])
        phot["DELTAMAG"] = phot["MODELMAG"] - phot["OBSMAG"]
        ########################################################################
        # Removing problematic lines
        phot = phot[np.isfinite(phot["DELTAMAG"])]
        dbs_dir = os.path.join(config["output_dir"],
                               "zps_{}".format(config["name"]))
        if not os.path.exists(dbs_dir):
            os.mkdir(dbs_dir)
        for band in context.bands:
            idx = np.where(band == phot["FILTER"])[0]
            if len(idx) == 0:
                continue
            outdb = os.path.join(dbs_dir, band)
            data = phot[idx]
            single_band_calib(data, outdb)



if __name__ == "__main__":
    main()

#     table = merge_zp_tables(redo=True)
