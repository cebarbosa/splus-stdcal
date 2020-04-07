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
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack, vstack
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib import cm
from tqdm import tqdm

import context
import misc

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

def comparison_sdss():
    """ Compares our zero points with those from SDSS. """
    ivezic_table = os.path.join(context.tables_dir,
                                "SDSS_S82_stars_Ivezic+07.fits")
    ivezic = Table.read(ivezic_table)
    tiles_dir = os.path.join(context.data_dir, "tiles")
    tiles = os.listdir(tiles_dir)
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    bands = ["G", "R", "I"]
    fig = plt.figure(figsize=(6.5, 3.3))
    outdir = os.path.join(context.plots_dir, "comparison_ivezic")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    tiles = [_ for _ in os.listdir(tiles_dir) if _.startswith("STRIPE")]
    for tile in tqdm(tiles):
        wdir = os.path.join(tiles_dir, tile)
        output = os.path.join(outdir, "{}.png".format(tile))
        for j, band in enumerate(bands):
            catfile = os.path.join(wdir,
                                   "{}_{}_single_catalog.fits".format(tile,
                                                                      band))
            if not os.path.exists(catfile):
                continue
            cat = Table.read(catfile)
            c0 = SkyCoord(ra=cat["ALPHA_J2000"] * u.degree,
                          dec=cat["DELTA_J2000"] * u.degree)
            c1 = SkyCoord(ra=ivezic["RAJ2000"], dec=ivezic["DEJ2000"])
            idx, d2d, d3d = c0.match_to_catalog_sky(c1)
            goodidx = np.where(d2d.to("arcsec").value < 1.)[0]
            splus = cat[goodidx]
            sdss = ivezic[idx][goodidx]
            diff = splus["MAG_AUTO"] - sdss["{}mag".format(band.lower())]
            diff = diff[sdss["rmag"] < 18]
            diff = diff[np.abs(diff) < 0.3]
            median = np.mean(diff)
            std = np.std(diff)
            ax = plt.subplot(1, 3, j + 1)
            ax.hist(diff, bins=31,
                    label=band.lower(),
                    color=colors[context.bands.index(band)],
                    range=(-0.3, 0.3))
            ax.axvline(x=median, c="k", ls="--",
                       label="median={:.2f} ({:.2f})".format(median, std))
            ax.set_xlabel("SPLUS - SDSS")
            ax.set_ylabel("frequency")
            plt.legend(prop={"size": 6})
        fig.suptitle(tile)
        plt.subplots_adjust(wspace=0.3, right=0.98, bottom=0.15, left=0.1)
        plt.savefig(output, dpi=250)
        plt.clf()

def compare_zps_kadu_laura():
    phots = ["FLUX_AUTO", "FLUX_APER"]
    wdir = "/home/kadu/Dropbox/SPLUS/stdcal/stdcal"
    zp_dir = os.path.join(wdir, "zeropoints/tiles")
    plots_dir = os.path.join(wdir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    zps = []
    stripe82 = [_ for _ in os.listdir(zp_dir) if _.startswith("STRIPE82")]
    for zptable in stripe82:
        zps.append(Table.read(os.path.join(zp_dir, zptable)))
    zps = vstack(zps)
    cmap = cm.get_cmap("Spectral_r")
    for phot in phots:
        data = zps[zps["FLUXTYPE"]==phot]
        means, stds, obands = [], [], []
        fig = plt.figure(figsize=(12,8))
        for i, band in enumerate(context.bands):
            idx = np.where(data["FILTER"]==band)[0]
            if len(idx) == 0:
                continue
            ax = plt.subplot(4, 3, i+1)
            c = cmap(i / 12.)
            d = data[idx]
            zp_laura, zperr_laura = [], []
            for tile in d["TILE"]:
                zp_laura.append(misc.get_zps_dr1(tile, [band])[0])
                zperr_laura.append(misc.get_zps_dr1(tile, [band],
                                                 field="sigZP")[0])
            zp_laura = np.array(zp_laura)
            zperr_laura = np.array(zperr_laura)
            diff = d["ZP"].data - zp_laura
            differr = np.sqrt(d["ZPERR"].data**2 + zperr_laura**2)
            weights = np.power(differr, -2)
            m = np.average(diff, weights=weights, axis=-1)
            sigm = np.sqrt(np.sum(np.power(weights * differr, 2), axis=-1))\
                      / \
                       np.sum(weights, axis=-1)
            std = np.std(diff)
            plt.errorbar(d["ZP"], diff, yerr=differr, fmt="o", c=c, label=band)
            plt.axhline(y=m, ls="--", c="k")
            plt.axhline(y=m + std,  ls=":", c="k")
            plt.axhline(y=m - std, ls=":", c="k")
            ax.legend()
            print(band, m, std)
            obands.append(band)
            means.append(m)
            stds.append(std)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                        right=False, which="both")
        plt.ylabel("$\Delta m_0$ (Kadu - Laura)", labelpad=15, size=20)
        plt.xlabel("mag", size=20, labelpad=5)
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.95,
                            wspace=0.13)
        plt.title(phot.replace("_", "\_"), fontdict = {'fontsize' : 20})
        plt.savefig(os.path.join(plots_dir, "zpdiff_bands-{}.png".format(
            phot)), dpi=200)
        plt.close()
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.errorbar(obands, means, yerr=stds, fmt="o")
        ax.axhline(y=0, ls="--", c="k")
        plt.ylabel("$\Delta m_0$ (Kadu - Laura)")
        plt.xlabel("Filter")
        plt.title(phot.replace("_", "\_"))
        plt.show()

if __name__ == "__main__":
    compare_zps_kadu_laura()
    # comparison_sdss()
    # std_star_gantt_chart()
    # compare_kappas()
    # compare_zps()
    # kappas_tololo()
    # Plot results from the zero point modeling
    # plot_zps_global("FLUX_AUTO")