# -*- coding: utf-8 -*-
""" 

Created on 01/12/17

Author : Carlos Eduardo Barbosa

Calibration of standard stars.

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, vstack, join
import speclite.filters
import pymc3

import context
import misc

def splus_mag_std_stars(stars, library="ctiostan"):
    """ Obtain magnitudes of """
    specdir = os.path.join(context.data_dir, "ststars", library)
    photsys = load_splus_filters()
    mags = {}
    for star in stars:
        specfile = os.path.join(specdir, "f{}.dat".format(star))
        if not os.path.exists(specfile):
            continue
        wave, flux = np.loadtxt(specfile, usecols=(0,1), unpack=True)
        wave = wave * u.AA
        flux = flux * (10 ** -16) * u.erg / u.cm / u.cm / u.s / u.AA
        m = photsys.get_ab_magnitudes(flux, wave, mask_invalid=True)
        mags[star] = m
    return mags

def load_splus_filters():
    """ Use speclite to load SPLUS filters for convolution. """
    filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "tables/filter_curves-master")
    ftable = ascii.read(os.path.join(context.tables_dir,
                                     "filter_lam_filename.txt"))
    filternames = []
    for f in ftable:
        fname = os.path.join(filters_dir, f["filename"])
        fdata = np.loadtxt(fname)
        fdata = np.vstack(((fdata[0,0]-1, 0), fdata, (fdata[-1,0]+1, 0)))
        w = fdata[:,0] * u.AA
        response = np.clip(fdata[:,1], 0., 1.)
        speclite.filters.FilterResponse(wavelength=w,
                  response=response, meta=dict(group_name="splus",
                  band_name=f["filter"]))
        filternames.append("splus-{}".format(f["filter"]))
    splus = speclite.filters.load_filters(*filternames)
    return splus

def model_zps(tab, flux_key="FLUX_AUTO", redo=False):
    """ Simple plot of the magnitude difference as a function of the airmass."""
    model = splus_mag_std_stars(stars)
    idx = sorted(np.arange(len(tab)), key=lambda i: tab["DATE"][i])
    tab = tab[idx]
    dates_str = "{}_{}".format(tab["DATE"][0], tab["DATE"][-1])
    for j, band in enumerate(context.bands):
        if j in [0,11]:
            continue
        # Getting data for a given filter
        idx = np.where(tab["FILTER"] == band)
        data = tab[idx]
        ########################################################################
        # Producing row with model magnitudes
        M = np.zeros(len(data), dtype=float)
        for star in set(data["STAR"]):
            idx = np.where(data["STAR"]==star)
            M[idx] = model[star]["splus-{}".format(band)]
        ########################################################################
        data["M"] = M
        data["m"] = misc.mag(np.array(data[flux_key]).T / data["EXPTIME"])
        # Producing observed magnitudes
        idx = np.isfinite(data["M"] - data["m"])
        data = data[idx]
        if len(data) == 0:
            continue
        bayesian_hierarchical_single_band(data, band, dates_str, redo=redo)

def bayesian_hierarchical_single_band(data, band, epoch, redo=False):
    """ Hierarchical Bayesian model with pooled data.

    Source: http://docs.pymc.io/notebooks/multilevel_modeling.html

    """
    outdir = os.path.join(context.home_dir, "zps", epoch)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = os.path.join(outdir,
                          "zp_{}_{}.fits".format(flux_key, band))
    outfile2 = os.path.join(outdir, "summary_{}_{}.csv".format(flux_key, band))
    if os.path.exists(outfile) and not redo:
        return
    data = data.to_pandas()
    nights = list(data.DATE.unique())
    nights_lookup = dict(zip(nights, range(len(nights))))
    data["night"] = data.DATE.copy()
    data['night'] = data.night.replace(nights_lookup).values
    with pymc3.Model():
        # Hyperpriors
        mu_a = pymc3.Normal('ZP', mu=20., sd=2)
        sigma_a = pymc3.HalfCauchy('ZPsigma', 5)
        mu_b = pymc3.Normal('Kappa', mu=0., sd=1)
        sigma_b = pymc3.HalfCauchy(r'Kappasigma', 5)
        # Randon intercepts
        zp = pymc3.Cauchy("zp", alpha=mu_a, beta=sigma_a, shape=len(
            nights))
        # Common slope
        kappa = pymc3.Cauchy(r"kappa", alpha=mu_b, beta=sigma_b, shape=len(
            nights))
        # Model error
        eps = pymc3.HalfCauchy(r"eps", 5)
        # Expected value
        linear_regress = zp[data["night"]] - kappa[data["night"]] * data[
            "AIRMASS"]
        pymc3.Cauchy('y', alpha=linear_regress, beta=eps,
                         observed=data["M"]-data["m"])
        trace = pymc3.sample()
        df = pymc3.stats.summary(trace)
        df.to_csv(outfile2)
    summary = []
    for night, idx in nights_lookup.items():
        t = Table([[night], [np.mean(trace["zp"][:, idx])],
                   [np.std(trace["zp"][:, idx])],
                   [np.mean(trace["kappa"][:, idx])],
                   [np.std(trace["kappa"][:, idx])]],
                  names=["date", "{}_zp".format(band),
                         "{}_zperr".format(band), "{}_kappa".format(
                          band), "{}_kappaerr".format(band)])
        summary.append(t)
    summary = vstack(summary)
    summary.write(outfile, overwrite=True, format="fits")
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


if __name__ == "__main__":
    phottable = os.path.join(context.home_dir, "tables/extmoni_phot.fits")
    phot = Table.read(phottable, format="fits")
    stars = [_.lower().replace(" ", "") for _ in phot["STAR"]]
    phot["STAR"] = stars
    stars = list(set(stars))
    nights = list(set(phot["DATE"]))
    ############################################################################
    # Cleaning sample using distance in the coordinate matching and Sextractor
    # flags
    phot = phot[phot["D2D"] < 10]
    phot = phot[phot["FLAGS"] == 0]
    ############################################################################
    # Removing problematic stars
    flagged_stars = ["ltt377", "ltt3218", "ltt4364", "cd-32_9927", "hr7950",
                     "hr4468"]
    badidx = []
    for fstar in flagged_stars:
        badidx.append(np.where(phot["STAR"]==fstar))
    badidx = np.hstack(badidx)[0]
    goodidx = np.arange(len(phot))
    goodidx = np.delete(goodidx, badidx)
    phot = phot[goodidx]
    ############################################################################
    sample_table = os.path.join(context.home_dir, "zps/sample.fits")
    print("Calibrating using a sample of {} stars.".format(len(phot)))
    phot.write(sample_table, overwrite=True)
    for flux_key in ["FLUX_AUTO", "FLUX_ISO", "FLUXAPERCOR"]:
        phot_old = phot[phot["DATE"] <= "2017-12-14"]
        phot_new = phot[phot["DATE"] > "2017-12-14"]
        model_zps(phot_old, redo=False, flux_key=flux_key)
        model_zps(phot_new, redo=False, flux_key=flux_key)
    table = merge_zp_tables(redo=True)
