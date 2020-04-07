# -*- coding: utf-8 -*-
""" 

Created on 22/02/19

Author : Carlos Eduardo Barbosa

Miscelaneous tasks used in and for the photometric calibration.

"""
from __future__ import print_function, division

import os
from datetime import datetime

import numpy as np
from astropy.table import Table, vstack, hstack

import context

def mag(f):
    return -2.5 * np.log10(f)

def table_DR_zps(directory):
    """ Produces a table containing the zero points from files produced for
    Data Releases. """
    filenames = os.listdir(directory)
    zps = []
    for fname in sorted(filenames):
        table = Table.read(os.path.join(directory, fname), format="ascii")
        zps.append(Table(table["ZP"], names=table["FILTER"]))
    filenames = Table([[f.split("_")[0] for f in filenames]], names=["FIELD"])
    zps = hstack([filenames, vstack(zps)])
    return zps

def select_nights(config):
    """ Select all nights in the configuration file range. """
    first_night = datetime.strptime(str(config["first_night"]), '%Y-%m-%d')
    last_night = datetime.strptime(str(config["last_night"]), '%Y-%m-%d')
    all_nights = sorted([_ for _ in sorted(os.listdir(config["singles_dir"])) \
                if os.path.isdir(os.path.join(config["singles_dir"], _))])
    nights = []
    for direc in all_nights:
        try:
            night = datetime.strptime(direc, "%Y-%m-%d")
        except:
            continue
        if first_night <= night <= last_night:
            nights.append(direc)
    return nights

def get_zps_dr1(tile, bands=None, field="ZP"):
    """ Read the table containing the zero points for a given tile and given
    bands. """
    bands = context.bands if bands is None else bands
    zpfile = os.path.join("/home/kadu/Dropbox/SPLUS/stdcal/stdcal",
                          "ZPfiles_Feb2019", "{}_ZP.cat".format(tile))
    zpdata = Table.read(zpfile, format="ascii")
    zpdict = dict([(t["FILTER"], t[field]) for t in zpdata])
    zps = np.array([zpdict[band] for band in bands])
    return zps


if __name__ == "__main__":
    zps = table_DR_zps("/home/kadu/Dropbox/SPLUS/stdcal/tables/ZPfiles_Feb2019")
    zps.write("/home/kadu/Dropbox/SPLUS/ifusci/tables/zps_Feb2019.fits",
              format="fits", overwrite=True)