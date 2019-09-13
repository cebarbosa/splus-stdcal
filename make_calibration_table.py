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
from astropy.table import Table, hstack
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

def find_bad_calibrations(tables, traces, output, nsig=5):
    """ Determine the maximum error allowed in the calibration and flag
    nights and bands that may have issues. """
    X = 2.0
    x = np.linspace(0, 0.4, 100)
    # Getting the model results
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0,1, 12)]
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
        ax = plt.subplot(4, 3, i + 1)
        if i < 9:
            ax.xaxis.set_ticklabels([])
        if band not in tables.keys():
            continue
        idxfile = os.path.join(wdir, "index_night_{}.yaml".format(band))
        if not os.path.exists(idxfile):
            continue
        with open(idxfile, "r") as f:
            ndct = yaml.load(f, Loader=yaml.FullLoader)
        nstars, zpx2err = [], []
        for night in nights:
            if night not in ndct.keys():
                zpx2err.append(np.infty)
                nstars.append(0)
                continue
            idx = ndct[night]
            zps = traces[band]["zp"][:, idx]
            ks = traces[band]["kappa"][:, idx]
            zpx2err.append(np.std(zps - ks * X))
            table = tables[band]
            idx = np.where(table["DATE"] == night)[0]
            table = table[idx]
            models = zps[:, None] - np.outer(ks, table["AIRMASS"])
            sigma = np.std(models, axis=0)
            mu = np.mean(models, axis=0)
            dm = np.abs(table["DELTAMAG"] - mu)
            nstars.append(len(np.where(dm <= 5 * sigma)[0]))
        zpx2err = np.array(zpx2err)
        nstars = np.array(nstars)
        mu, median, sd = sigma_clipped_stats(zpx2err[np.isfinite(zpx2err)])
        maxerr = mu + nsig * sd
        flag = np.where(zpx2err < maxerr, 1, 0) * nstars
        t = Table([flag], names= ["NSTARS_{}".format(band)])
        results.append(t)
        ax.hist(zpx2err, bins=np.linspace(0, 0.4, 20), color=colors[i],
                density=True, label=band)
        ax.plot(x, norm.pdf(x, mu, sd), label="model")
        ax.axvline(x=maxerr, ls="--", label="$\sigma_{{zp}}={0:.2}$".format(
            maxerr))
        leg = ax.legend(prop={'size': 8}, loc=1)
    results = hstack(results)
    print(results)
    plt.subplots_adjust(left=0.07, right=0.98, hspace=0.05, top=0.95,
                        bottom=0.08, wspace=0.2)
    plt.savefig(output)
    plt.show()
    plt.clf()
    return

if __name__ == "__main__":
    config_files = ["config_mode0.yaml", "config_mode5.yaml"]
    phots = ["FLUX_APER", "FLUX_AUTO"]
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for phot in phots:
            wdir = os.path.join(config["output_dir"], "zps-{}-{}".format(
                config["name"], phot))
            tables, traces = {}, {}
            for band in context.bands:
                filename = os.path.join(wdir, "phot_{}.fits".format(band))
                if not os.path.exists(filename):
                    continue
                tables[band] = Table.read(filename, format="fits")
                db = os.path.join(wdir, band)
                traces[band] = load_traces(db)
            outimg = os.path.join(wdir, "maxerr-{}-{}.pdf".format(
                                  config["name"], phot))
            fit_flags = find_bad_calibrations(tables, traces, outimg)