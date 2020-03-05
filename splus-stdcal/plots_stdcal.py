# -*- coding: utf-8 -*-
""" 

Created on 26/02/19

Author : Carlos Eduardo Barbosa

Plots related to the modeling of zero points obtained with stdcal.py.

"""
from __future__ import print_function, division

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.table import Table, hstack

import context
import misc

def tiles_std_zps(fluxkey, redo=False):
    """ Merge tables of zero points of different bands into one table. """
    wdir = os.path.join(context.data_dir, "tiles_zps")
    output = os.path.join(context.home_dir, "tables/zps_tiles_{}.fits".format(
        fluxkey))
    if os.path.exists(output) and not redo:
        return Table.read(output)
    filenames = os.listdir(wdir)
    tiles = Table([list(set([_.split("_")[0] for _ in filenames]))],
                  names=["FIELD"])
    zps = np.zeros((len(tiles), len(context.bands))) * np.nan
    zperrs = np.full_like(zps, np.nan)
    for j, tile in enumerate(tiles["FIELD"]):
        tables = [_ for _ in filenames if _.startswith(tile)]
        for table in tables:
            field, band, cattype = table.split("_")
            i = context.bands.index(band)
            data = Table.read(os.path.join(wdir, table))
            if fluxkey not in data["phot"]:
                continue
            data = data[np.where(data["phot"] == fluxkey)]
            zps[j, i] = data["zp"]
            zperrs[j, i] = data["zperr"]
    zps = Table(zps, names=context.bands)
    zperrs = Table(zps, names=["{}_err".format(_) for _ in context.bands])
    results = hstack([tiles, zps, zperrs])
    results.write(output, overwrite=True)
    return results

def plot_zps_hist(data, output, redo=False):
    """ Plot a histogram of the zeropoints for each band. """
    # Producing plots
    cmap = cm.get_cmap("Spectral_r")
    colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    fig = plt.figure(1, figsize=(10,8))
    for i, band in enumerate(context.bands):
        bdata = data[band]
        bdata = bdata[~np.isnan(bdata)]
        if len(bdata) == 0:
            continue
        ax = plt.subplot(3,4,i+1)
        ax.minorticks_on()
        mu = np.mean(bdata)
        std = np.std(bdata)
        ax.hist(bdata, color=colors[i], label="$\mu=${:.2f} $\sigma=${"
                                             ":.2f}".format(mu, std))
        ax.set_xlabel("zero point (mag)")
        ax.legend(prop={"size":8}, title=band.lower().replace("f", "F"))
    plt.show()
    plt.clf()
    return

def plot_colors_comparison(data, refdata, output, redo=False):
    """ Compare the zero point colors. """
    if os.path.exists(output) and not redo:
        return
    fig = plt.figure(1, figsize=(8,6))
    colors = get_colors()
    for i, (c1, c2) in enumerate(colors):
        color_str = "{}-{}".format(c1, c2).lower().replace("f", "F")
        diff = (data[c1] - data[c2]) - (refdata[c1] - refdata[c2])
        mu = np.nanmean(diff)
        std = np.nanstd(diff)
        ax = plt.subplot(4, 2, i+1)
        ax.hist(diff, bins=30,
                label="$\mu=${:.2f} $\sigma=${:.2f}".format(mu, std))
        ax.legend(prop={"size": 8}, title=color_str)
        ax.set_xlabel("$\Delta$ color")
        ax.set_ylabel("\#")
    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08,
                        hspace=0.4)
    plt.savefig(output, dpi=200)
    plt.clf()
    return

def get_colors(nc2=None, bc2=None):
    """ Define colors between broad and narrow bands to be used in the plots."""
    nc2 = "F430" if nc2 is None else nc2
    bc2 = "R" if bc2 is None else bc2
    nc1 = context.narrow_bands[:]
    nc1.remove(nc2)
    colors_narrow = np.array([nc1, [nc2] * len(nc1)]).T
    bc1 = context.broad_bands[:]
    bc1.remove(bc2)
    # Remove bands without data
    bc1.remove("U")
    bc1.remove("Z")
    colors_broad = np.array([bc1, len(bc1) * [bc2]]).T
    colors = np.vstack([colors_narrow, colors_broad])
    sorted_colors = []
    for i, (c1, c2) in enumerate(colors):
        if context.bands.index(c1) > context.bands.index(c2):
            c1, c2 = c2, c1
        sorted_colors.append([c1, c2])
    return sorted_colors

def make_table():
    tabfile = os.path.join(context.home_dir, "zeropoint.fits")
    tabdata = Table.read(tabfile, format="fits")
    pairs = np.array(tabdata.keys())[1:].reshape((-1, 4))
    cols1 = [Table([tabdata["date"]])]
    cols2 = [Table([tabdata["date"]])]
    names = ["Date"]
    for c1, c2, c3, c4 in pairs:
        col1 = []
        col2 = []
        for i in range(len(tabdata)):
            col1.append("${:S}$".format(ufloat(tabdata[c1][i], tabdata[c2][
                i])))
            col2.append("${:S}$".format(ufloat(tabdata[c3][i], tabdata[c4][
                i])))
        col1 = Table([col1], names=[c1])
        col2 = Table([col2], names=[c3])
        cols1.append(col1)
        cols2.append(col2)
    cols1 = hstack(cols1)
    cols2 = hstack(cols2)
    output1 = os.path.join(context.tables_dir, "zps.tex")
    output2 = os.path.join(context.tables_dir, "kappas.tex")
    cols1.write(output1, format="latex", overwrite=True)
    cols2.write(output2, format="latex", overwrite=True)

if __name__ == "__main__":
    for flux_key in ["FLUX_AUTO", "FLUX_ISO", "FLUXAPERCOR"]:
        print(flux_key)
        # plot_nights(flux_key)
        stddata = tiles_std_zps(flux_key, redo=True)
        # Read table from Fábio containing only photometric fields
        # goodfields_file = os.path.join(context.tables_dir,
        #                                      "stripe82_photometric_fields.txt")
        # goodfields = Table.read(goodfields_file, format="ascii")
        # idx = [i for i,field in enumerate(mydata["FIELD"]) if field in
        #               goodfields["field"]]
        # mydata = mydata[idx]
        ########################################################################
        # Zero points from Laura and Alberto
        datas82 = misc.table_DR_zps(os.path.join(context.tables_dir,
                                                 "ZPfiles_Feb2019"))
        ########################################################################
        # Align my table with those from L&A
        fields = np.intersect1d(stddata["FIELD"].data, datas82["FIELD"].data)
        idx = [np.where(stddata["FIELD"] == field)[0][0] for field in fields]
        stddata = stddata[idx]
        idx = [np.where(datas82["FIELD"] == field)[0][0] for field in fields]
        datas82 = datas82[idx]
        ########################################################################
        # Producing a histogram with zero point values
        # out = os.path.join(context.plots_dir,
        #                    "hist_zps_{}.png".format(flux_key))
        # plot_zps_hist(stddata, out, redo=True)
        ########################################################################
        # Comparing the colors
        # out = os.path.join(context.plots_dir,
        #                    "hist_colors_{}.png".format(flux_key))
        # plot_colors_comparison(stddata, datas82, out)
        #######################################################################