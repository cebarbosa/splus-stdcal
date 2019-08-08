# -*- coding: utf-8 -*-
""" 

Created on 12/03/19

Author : Carlos Eduardo Barbosa

Compares the calibration of the broadbands with those from Ivezic using SDSS.
"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import context

if __name__ == "__main__":
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
                             "{}_{}_single_catalog.fits".format(tile, band))
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
            diff = diff[np.abs(diff)<0.3]
            median = np.mean(diff)
            std = np.std(diff)
            ax = plt.subplot(1,3, j+1)
            ax.hist(diff, bins=31,
                     label=band.lower(),
                     color=colors[context.bands.index(band)],
                    range=(-0.3, 0.3))
            ax.axvline(x=median, c="k", ls="--",
                       label="median={:.2f} ({:.2f})".format(median, std))
            ax.set_xlabel("SPLUS - SDSS")
            ax.set_ylabel("frequency")
            plt.legend(prop={"size":6})
        fig.suptitle(tile)
        plt.subplots_adjust(wspace=0.3, right=0.98, bottom=0.15, left=0.1)
        plt.savefig(output, dpi=250)
        plt.clf()