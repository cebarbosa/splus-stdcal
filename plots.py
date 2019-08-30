# -*- coding: utf-8 -*-
""" 

Created on 30/01/18

Author : Carlos Eduardo Barbosa

Produce a few plots for the report on the photometric calibration.
"""

from __future__ import print_function, division

import os
import yaml
import pickle
from datetime import datetime

import numpy as np
from astropy.table import Table, hstack
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import date2num
from matplotlib import cm
from tqdm import tqdm

import context

def std_star_gantt_chart():
    """ Plot the observation dates of standard stars. """
    table = Table.read("/home/kadu/Dropbox/SPLUS/stdcal/tables/"
                       "extmoni_phot.fits")
    stars = np.unique(table["STAR"])
    mags = splus_mag_std_stars(stars)
    print(mags)
    R = [np.round(float(mags[_]["splus-R"]), 2) for _ in stars]
    F861 = [np.round(float(mags[_]["splus-F861"]), 2) for _ in stars]
    idx = np.argsort(R)[::-1]
    stars = stars[idx]
    R = np.array(R)[idx]
    F861 = np.array(F861)[idx]
    print(zip(stars, R))
    print("There are {} different stars observed for calibration.".format(
           len(stars)))
    fig = plt.figure(figsize=(8,4))
    ax = plt.subplot(111)
    ax.minorticks_on()
    ys = []
    for i, star in enumerate(stars):
        idx = np.where(table["STAR"] == star)
        dates = np.unique(table[idx]["DATE"]).tolist()
        dates = date2num([datetime.strptime(_, "%Y-%m-%d") for _ in dates])
        y = np.ones(len(dates)) * i
        ys.append(i)
        ax.plot_date(dates, y, "x", ms=10)
    for s in ["ltt", "hr", "feige"]:
        stars = [_.replace(s, "{} ".format(s)) for _ in stars]
    stars = [_.upper() for _ in stars]
    plt.yticks(ys, stars)
    ylim = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim(ylim)
    plt.yticks(ys, F861)
    ax.set_xlabel("Observing date")
    ax.set_ylabel("Standard star")
    ax2.set_ylabel("F861")
    ax3 = ax.twinx()
    plt.yticks(ys, R)
    ax3.set_ylabel("R")
    ax3.spines['right'].set_position(('axes', 1.15))
    ax3.set_ylim(ylim)
    plt.subplots_adjust(left=0.12, right=0.82, bottom=0.09, top=0.98)
    ax.tick_params(axis='y', which='minor', left='off', right="off")
    plt.savefig("/home/kadu/Dropbox/SPLUS/stdcal/figs/gantt_chart.png",
                dpi=150)
    return

def compare_kappas():
    """ Compare the extinction coefficients with results from CEFCA. """
    fdata = ascii.read(os.path.join(context.tables_dir, "filter_lam.txt"))
    fdata = fdata[np.argsort(fdata["lam(AA)"])]
    jplus = Table.read(os.path.join(context.tables_dir, "jplus_kappas.txt"),
                       format="ascii")
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0,1, 12)]
    j = 1
    plt.style.use("ggplot")
    for i, (band, lam) in enumerate(fdata):
        dbname = os.path.join(context.home_dir, "sbcal", "{}.pkl".format(band))
        if not os.path.exists(dbname):
            continue
        with open(dbname, 'rb') as buff:
            mcmc = pickle.load(buff)
        trace = mcmc["trace"]
        ax = plt.subplot(2, 5, j)
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both")
        ax.hist(trace["Kappa"], color=colors[i], normed=True, label=band)
        idx = np.argwhere(jplus["band"] == band)
        ax.axvline(jplus["kappa"][idx], c="k", ls="--")
        ax.legend()
        ax.set_xlabel("$\kappa$")
        if i in [1,6]:
            ax.set_ylabel("Frequency")
        j += 1
    output = os.path.join(context.plots_dir, "kappas_comparison.png")
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
    plt.savefig(output, dpi=200)
    plt.show()
    plt.clf()

def kappas_tololo():
    """ Compare kappas with results from Stone and Baldwin 1983. """
    fdata = ascii.read(os.path.join(context.tables_dir, "filter_lam.txt"))
    fdata = fdata[np.argsort(fdata["lam(AA)"])]
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0,1, 12)]
    sb83 = ascii.read((os.path.join(context.tables_dir,
                                    "kappas_stone_baldwin_1983.txt")))
    # plt.style.use("ggplot")
    plt.figure(1, figsize=(3.32, 2.5))
    ax = plt.subplot(111)
    ax.minorticks_on()
    ax.plot(sb83["col1"], sb83["col2"], "s",
                  label = "Stone \& Baldwin 1983", mec="C0", c="w")
    legs = []
    for i, (band, lam) in enumerate(fdata):
        dbname = os.path.join(context.home_dir, "sbcal", "{}.pkl".format(band))
        if not os.path.exists(dbname):
            continue
        with open(dbname, 'rb') as buff:
            mcmc = pickle.load(buff)
        trace = mcmc["trace"]
        kappas = trace["Kappa"]
        kappa = kappas.mean()
        kappa5 = np.percentile(kappas, 5)
        kappa95 = np.percentile(kappas, 95)
        error = np.atleast_2d([kappa95 - kappa, kappa - kappa5]).T
        ax.errorbar(lam, kappa, yerr = error,
                    fmt="o", ecolor="0.5", c=colors[i],
                    label=context.bands_names[band])
        legs.append(context.bands_names[band])
    lines = ax.get_lines()
    anchorxy1 = np.array([0.4,0.46])
    anchorxy2 = np.array([0.4,0.84])
    l1 = ax.legend(lines[1:], legs, ncol=2,
                   loc="lower left", bbox_to_anchor=anchorxy1,
                   frameon=False, prop={'size': 8})
    l2 = ax.legend(lines[:1], ["Stone \& Baldwin 1983"], frameon=False,
                   loc="lower left", bbox_to_anchor=anchorxy2,
                   prop={'size': 8})
    ax.add_artist(l1)
    ax.add_artist(l2)
    ax.set_xlabel("Wavelength [\\r{A}]")
    ax.set_ylabel("$\kappa (\lambda)$ [mag airmass$^{\\rm -1}$]")
    ax.set_ylim(-0.02, 0.6)
    ax.set_xlim(3000, 9000)
    plt.subplots_adjust(left=0.135, bottom=0.155, top=0.97, right=0.96)
    plt.savefig(os.path.join(context.plots_dir, "kappas_splus.png"), dpi=250)
    plt.clf()
    return

def plot_zps_nights(config, fkey, redo=False, alpha=15.865):
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
    if config["name"] == "mode0":
        X = np.linspace(1., 1.4, 100)
    else:
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
                leg = ax.legend(prop = {'size': 5})
                leg.set_title(band, prop={"size": 5})
                if i < 9:
                    ax.xaxis.set_ticklabels([])
            plt.subplots_adjust(left=0.07, right=0.98, hspace=0.05, top=0.95,
                                bottom=0.08, wspace=0.2)
            pdf.savefig()
            plt.close()

def load_traces(db, npop=None, alpha=15.865):
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
    # maps, errors = [], []
    # for comp in range(len(npop)):
    #     nssps = npop[comp]
    #     for param in params:
    #         if param in ["Av"]:
    #             trace = np.vstack([data[num]["Av_{}".format(comp)]
    #                                       for num in range(ntraces)])
    #         else:
    #             ws, vs = [], []
    #             for nssp in range(nssps):
    #                 v = np.vstack([data[num]["{}_{}_{}".format(param, comp, nssp)]
    #                                for num in range(ntraces)])
    #                 w = np.vstack([data[num]["flux_{}_{}".format(comp, nssp)]
    #                                for num in range(ntraces)])
    #                 vs.append(v)
    #                 ws.append(w)
    #             vs = np.array(vs)
    #             ws = np.array(ws)
    #             trace = np.average(vs, weights=ws, axis=0)
    #         med = np.median(trace, axis=0)
    #         lower = med - np.percentile(trace, alpha, axis=0)
    #         upper =  np.percentile(trace, 100 - alpha, axis=0) - med
    #         maps.append(med)
    #         errors.append(np.column_stack((lower, upper)))
    # maps = Table(maps, names=params)
    # return maps, errors

def plot_zps_global(flux_key):
    """ Produces plot with the pooled results. """
    wdir = os.path.join(context.home_dir, "zps")
    stars = Table.read(os.path.join(wdir, "sample.fits"))
    stdmags = splus_mag_std_stars(set(stars["STAR"]))
    fig = plt.figure(1, figsize=(8,4))
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0,1, 12)]
    for i, band in enumerate(context.bands):
        if band in ["U", "Z"]:
            continue
        ax = plt.subplot(4,3, i+1)
        data = stars[stars["FILTER"]==band]
        # Sort according to date
        idx = sorted(range(len(data)), key=lambda k: data["DATE"][k])
        data = data[idx]
        m = misc.mag(data[flux_key])
        M = np.zeros(len(data))
        for star in set(data["STAR"]):
            idx = np.where(data["STAR"] == star)
            M[idx] = stdmags[star]["splus-{}".format(band)]
        idx_before = data["DATE"] <= "2017-12-14"
        idx_after = data["DATE"] > "2017-12-14"
        labels = ["$<=$ 2017-12-14", "$>$ 2017-12-14"]

        for j, idx in enumerate([idx_before, idx_after]):
            ax.plot(data["AIRMASS"][idx], M[idx] - m[idx], "o",
                    label=labels[j])
        # ax.plot(data["AIRMASS"], M - m, "o", c=colors[i],
        #         label="N={}".format(len(M)))
        ax.set_xlabel("X")
        ax.set_ylabel("M - m")
        plt.legend(title=band.lower().replace("f", "F"))
    plt.show()
    return

def compare_zps(flux_key="FLUX_AUTO", update=False):
    zpsla = Table.read(os.path.join(context.tables_dir, "ZP_finals.cat"),
                          format="ascii")
    zptable = os.path.join(os.path.join(context.tables_dir,
                                        "zps_tiles_{}.fits".format(flux_key)))
    if not os.path.exists(zptable) or update:
        zp_files_dir = os.path.join(os.path.join(context.data_dir, "tiles_zps"))
        filenames = os.listdir(zp_files_dir)
        fields = sorted(set([_.split("_")[0] for _ in filenames]))
        zps = []
        for field in fields:
            zpsfield = np.zeros(12) * np.nan
            for i, band in enumerate(context.bands):
                fname = os.path.join(zp_files_dir, "{}_{}_single.fits".format(
                    field, band))
                if not os.path.exists(fname):
                    continue
                t = Table.read(fname, format="fits")
                idx = np.where(t["phot"]==flux_key)
                zpsfield[i] = t["zp"][idx]
            zps.append(zpsfield)
        zps = np.array(zps)
        zps = Table(zps, names=context.bands)
        fields = Table([fields], names=["Field"])
        zps = hstack([fields, zps])
        zps.write(zptable, overwrite=True)
    zps = Table.read(zptable, format="fits")
    zps.rename_column("FIELD", "Field")
    fields = np.intersect1d(zpsla["Field"].data, zps["Field"].data)
    idx = [np.where(zps["Field"] == field)[0][0] for field in fields]
    zps = zps[idx]
    idx = [np.where(zpsla["Field"] == field)[0][0] for field in fields]
    zpsla = zpsla[idx]
    ax = plt.subplot(1,1,1)
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    for i,band in enumerate(context.bands):
        if i in [0, 11]:
            continue
        diff = zps[band] - zpsla["ZP_{}".format(band).replace("F", "")]
        plt.errorbar(i, np.nanmean(diff), yerr=np.nanstd(diff), c=colors[i],
                     fmt="o")
    plt.ylim(-.2, .5)
    plt.xlim(0, 11)
    plt.xticks(np.arange(12), context.bands)
    plt.axhline(y=0, ls="--", c="k")
    plt.ylabel("$\Delta$ zp")
    plt.xlabel("Filter")
    plt.title("Comparison for {} tiles.".format(len(zps)))
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.92)
    plt.savefig(os.path.join(context.plots_dir, "comparison_laura.png"))

if __name__ == "__main__":
    config_files = ["config_mode5.yaml", "config_mode0.yaml"]
    for filename in config_files:
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for flux in ["FLUX_APER", "FLUX_AUTO"]:
            plot_zps_nights(config, flux)

    # std_star_gantt_chart()
    # compare_kappas()
    # compare_zps()
    # kappas_tololo()
    # Plot results from the zero point modeling
    # plot_zps_global("FLUX_AUTO")