# -*- coding: utf-8 -*-
""" 

Created on 03/09/19

Author : Carlos Eduardo Barbosa

Read results from stdcal and selects the nights that can be used for
calibration according to a set of criteria

"""
from __future__ import print_function, division

import os
import yaml

import numpy as np
from astropy.table import Table, hstack, vstack, join
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from astropy.stats import sigma_clipped_stats


import context

def load_traces(db):
    params = ["zp", "kappa"]
    if not os.path.exists(db):
        return None
    ntraces = len(os.listdir(db))
    data = [np.load(os.path.join(db, _, "samples.npz")) for _ in
            os.listdir(db)]
    traces = {}
    for param in params:
        traces[param] = np.vstack([data[num][param] for num in range(ntraces)])
    return traces

def make_calibration_table(config, wdir, nsig=3, X=2.0):
    """ Determine the maximum error allowed in the calibration and flag
    nights and bands that may have issues. """
    phot = os.path.split(wdir)[-1]
    outimg = os.path.join(wdir, "maxerr.pdf")
    x = np.linspace(0, 0.4, 100)
    # Getting the model results
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0,1, 12)]
    ############################################################################
    # Load all tables prior to loop to get unique list of nights
    tables = {}
    for band in context.bands:
        filename = os.path.join(wdir, "phot_{}.fits".format(band))
        if not os.path.exists(filename):
            continue
        tables[band] = Table.read(filename, format="fits")
    nights = [list(set(tables[band]["DATE"])) for band in \
              context.bands if band in tables.keys()]
    nights = np.unique(np.hstack(nights))
    fig = plt.figure(figsize=(8, 5))
    title = "{} {}".format(config["name"].upper(),
                              phot.replace("_", "\_"))
    fig.text(0.5, 0.97, title, ha='center')
    fig.text(0.5, 0.02, '$\sigma_{zp} (X=2)$', ha='center')
    fig.text(0.01, 0.5, 'density (normalized)', va='center',
             rotation='vertical')
    results = [Table([nights], names=["DATE"])]
    for i, band in enumerate(context.bands):
        if band not in tables:
            continue
        table = tables[band]
        db = os.path.join(wdir, band)
        trace = load_traces(db)
        ax = plt.subplot(4, 3, i + 1)
        if i < 9:
            ax.xaxis.set_ticklabels([])
        idxfile = os.path.join(wdir, "index_night_{}.yaml".format(band))
        if not os.path.exists(idxfile):
            continue
        with open(idxfile, "r") as f:
            ndct = yaml.load(f)
        nin, zpx2err, ntot = [], [], []
        zeropoints, zeropoints_err, kappas, kappas_err = [], [], [], []
        for night in nights:
            if night not in ndct.keys():
                zpx2err.append(np.infty)
                nin.append(0)
                ntot.append(0)
                zeropoints.append(np.nan)
                zeropoints_err.append(np.nan)
                kappas.append(np.nan)
                kappas_err.append(np.nan)
                continue
            idx = ndct[night]
            zps = trace["zp"][:, idx]
            ks = trace["kappa"][:, idx]
            zpx2err.append(np.std(zps - ks * X))
            idx = np.where(table["DATE"] == night)[0]
            table = table[idx]
            models = zps[:, None] - np.outer(ks, table["AIRMASS"])
            sigma = np.std(models, axis=0)
            mu = np.mean(models, axis=0)
            dm = np.abs(table["DELTAMAG"] - mu)
            n = len(np.where(dm <= nsig * sigma)[0])
            nin.append(n)
            ntot.append(len(dm))
            zeropoints.append(np.mean(zps))
            zeropoints_err.append(np.std(zps))
            kappas.append(np.mean(ks))
            kappas_err.append(np.std(ks))
        zpx2err = np.array(zpx2err)
        nin = np.array(nin)
        mu, median, sd = sigma_clipped_stats(zpx2err[np.isfinite(zpx2err)])
        maxerr = mu + nsig * sd
        nin = np.where(zpx2err < maxerr, 1, 0) * nin
        ntot = np.array(ntot)
        names = ["ZP", "eZP", "K", "eK", "N", "Ntot"]
        t = Table([zeropoints, zeropoints_err, kappas, kappas_err, nin, ntot],
                  names= ["{}_{}".format(n, band) for n in names])
        results.append(t)
        ax.hist(zpx2err, bins=np.linspace(0, 0.4, 20), color=colors[i],
                density=True, label=band)
        ax.plot(x, norm.pdf(x, mu, sd), label="model")
        ax.axvline(x=maxerr, ls="--", label="$\sigma_{{zp}}={0:.2}$".format(
            maxerr))
        leg = ax.legend(prop={'size': 8}, loc=1)
    results = hstack(results)
    plt.subplots_adjust(left=0.07, right=0.98, hspace=0.05, top=0.95,
                        bottom=0.08, wspace=0.2)
    plt.savefig(outimg)
    plt.clf()
    for fmt in ["fits", "csv"]:
        outtable = os.path.join(wdir, "zps-dates.{}".format(fmt))
        print(outtable)
        results.write(outtable, overwrite=True)
    return

def main():
    config_files = ["config_mode0.yaml", "config_mode5.yaml"]
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.load(f)
        phots = config["sex_phot"]
        calib_dir = os.path.join(config["main_dir"], "calib")
        for phot in phots:
            calibs = [_ for _ in os.listdir(calib_dir) if _.startswith(
                config["name"])]
            for calib in calibs:
                wdir = os.path.join(calib_dir, calib, phot)
                make_calibration_table(config, wdir)

if __name__ == "__main__":
    main()