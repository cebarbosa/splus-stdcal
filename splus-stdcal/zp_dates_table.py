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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm


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

def zp_dates_table(config, wdir, nsig=5, X=2.0):
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
    ############################################################################
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
            if night not in ndct:
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
            zeropoints.append(np.mean(zps))
            zeropoints_err.append(np.std(zps))
            kappas.append(np.mean(ks))
            kappas_err.append(np.std(ks))
            ntot.append(len(idx))
            # Calculates the number of inliers
            date = table[idx]
            models = zps[:, None] - np.outer(ks, date["AIRMASS"].data)
            sigma = np.std(models, axis=0)
            mu = np.mean(models, axis=0)
            dm = np.abs(date["DELTAMAG"].data - mu)
            nin.append(len(np.where(dm <= nsig * sigma)[0]))
        zpx2err = np.array(zpx2err)
        ntot = np.array(ntot)
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
        results.write(outtable, overwrite=True)
    return

def plot_zps_dates(config, fkey, alpha=15.865):
    """ Plot results for modeling for individual nights  """
    wdir = os.path.join(config["output_dir"], "zps-{}-{}".format(
                        config["name"], fkey))
    ########################################################################
    # Getting the model results
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0,1, 12)]
    fig = plt.figure(1, figsize=(6.5, 4))
    # Read all tables and see what nights should be processed
    # Also, read all traces to be used in plots
    tables, traces = {}, {}
    for band in context.bands:
        filename = os.path.join(wdir, "phot_{}.fits".format(band))
        if not os.path.exists(filename):
            continue
        tables[band] = Table.read(filename, format="fits")
        db = os.path.join(wdir, band)
        traces[band] = load_traces(db)
    nights = [list(set(tables[band]["DATE"])) for band in \
            context.bands if band in tables.keys()]
    nights = np.unique(np.hstack(nights))
    X = np.linspace(1., 1.8, 100)
    output = os.path.join(wdir, "results-{}-{}.pdf".format(config["name"],
                                                           fkey))
    with PdfPages(output) as pdf:
        for night in tqdm(nights):
            fig, ax = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True,
                                   figsize=(8, 5))
            fig.text(0.5, 0.03, 'X', ha='center')
            fig.text(0.01, 0.5, '$M - m$', va='center', rotation='vertical')
            fig.text(0.5, 0.97, night, ha='center')
            plt.title("Date: {}".format(night))
            for i, band in enumerate(context.bands):
                if band not in tables.keys():
                    continue
                ax = plt.subplot(4, 3, i + 1)
                # Plot table data
                data = tables[band]
                ax.plot(data["AIRMASS"], data["MODELMAG"] - data["OBSMAG"], ".",
                        c="0.9", label="All")
                idx = np.where(data["DATE"]==night)[0]
                stars = set(data["STAR"][idx])
                symbols = ["o", "s", "^", "x", ""]
                for j,star in enumerate(stars):
                    idx = np.where((data["DATE"] == night) &
                                   (data["STAR"]==star))[0]
                    ax.plot(data["AIRMASS"][idx],
                            data["MODELMAG"][idx] - data["OBSMAG"][idx],
                            marker=symbols[j], c=colors[i],
                            label=star.upper().replace("_", "\_"),
                            ls="none")
                # Plot results based on traces
                idxfile = os.path.join(wdir, "index_night_{}.yaml".format(band))
                if not os.path.exists(idxfile):
                    continue
                with open(idxfile, "r") as f:
                    ndct = yaml.load(f, Loader=yaml.FullLoader)
                if night not in ndct.keys():
                    continue
                idx = ndct[night]
                zps = traces[band]["zp"][:,idx]
                ks = traces[band]["kappa"][:,idx]
                y = zps[:, None] - np.outer(ks, X)
                ylower = np.percentile(y, alpha, axis=0)
                yupper = np.percentile(y, 100 - alpha, axis=0)
                ax.plot(X, zps.mean() - X * ks.mean(), "-", c=colors[i],
                        label="Model")
                ax.fill_between(X, ylower, yupper, color=colors[i], alpha=0.5)
                leg = ax.legend(prop = {'size': 5}, loc=4)
                leg.set_title(band, prop={"size": 5})
                if i < 9:
                    ax.xaxis.set_ticklabels([])
            plt.subplots_adjust(left=0.07, right=0.98, hspace=0.05, top=0.95,
                                bottom=0.08, wspace=0.2)
            pdf.savefig()
            plt.close()

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
                zp_dates_table(config, wdir)
                plot_zps_dates(config, phot)

if __name__ == "__main__":
    main()