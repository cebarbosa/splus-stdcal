# -*- coding: utf-8 -*-
""" 

Created on 13/09/19

Author : Carlos Eduardo Barbosa

Read photometric tables to determine which stars and bands are saturated.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.table import Table, vstack, hstack
import matplotlib.pyplot as plt

import context

def sextractor_flags(x):
    powers = []
    i = 1
    while i <= x:
        if i & x:
            powers.append(i)
        i <<= 1
    return powers

if __name__ == "__main__":
    wdir = "/home/kadu/Dropbox/SPLUS/stdcal/stdcal"
    tablenames = ["phottable_mode0.fits", "phottable_mode5.fits"]
    tabledata = []
    target_flag = 4 # Saturation flag
    for table in tablenames:
        t = Table.read(os.path.join(wdir, table), format="fits")
        tabledata.append(t)
    tabledata = vstack(tabledata)
    stars = set(tabledata["STAR"])
    fracs = []
    for star in stars:
        idx = np.where(tabledata["STAR"] == star)[0]
        stardata = tabledata[idx]
        starfrac = []
        for band in context.bands:
            idx = np.where(stardata["FILTER"]==band)[0]
            data = stardata[idx].as_array()
            star_band_flags = data["FLAGS"]
            nsat = 0
            for flag in set(star_band_flags):
                if flag < target_flag:
                    continue
                powers = sextractor_flags(flag)
                if target_flag in powers:
                    nobs = len(np.where(star_band_flags == flag)[0])
                    nsat += nobs
            frac = np.round(nsat / len(data), 2)
            starfrac.append(frac)
        fracs.append(starfrac)
    fracs = Table(np.array(fracs), names=context.bands)
    stars = Table([list(stars)], names=["STAR"])
    outtable = hstack([stars, fracs])
    outtable.write(os.path.join(wdir, "saturation_fraction.csv"),
                     format="ascii.csv")



