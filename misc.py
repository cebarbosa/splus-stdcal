# -*- coding: utf-8 -*-
""" 

Created on 22/02/19

Author : Carlos Eduardo Barbosa

Miscelaneous tasks used in and for the photometric calibration.

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
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

def get_apertures(logfile):
    """ Parses the logfile to obtain the apertures used in the photometry."""
    splitter = "(pixels, diameter)"
    with open(logfile) as f:
        data = [_ for _ in f.readlines() if splitter in _][0]
    data = data.split(splitter)[1].replace("(", "").replace(")", "")
    data = np.array([float(_) for _ in data.split(",")])
    radius = np.round(0.5 * data * context.ps * u.pixel, 3)
    table = Table([np.arange(len(radius)), radius], names=["aperture",
                                                           "radius"])
    return table

# def get_apertures(logfile, napers):
#     """ Parses the logfile to obtain the apertures used in the photometry."""
#     splitter = "(pixels, diameter)"
#     with open(logfile) as f:
#         data = [_ for _ in f.readlines() if splitter in _]
#     data = [d.split(splitter)[1].replace("(", "").replace(")", "") for d in
#             data]
#     data = [np.array([float(_) for _ in d.split(",")]) for d in data]
#     sizes = [len(_) for _ in data]
#     idx = sizes.index(napers)
#     radius = np.round(0.5 * data[idx] * context.ps * u.pixel, 3)
#     table = Table([np.arange(len(radius)), radius], names=["aperture",
#                                                            "radius"])
#     return table


if __name__ == "__main__":
    zps = table_DR_zps("/home/kadu/Dropbox/SPLUS/stdcal/tables/ZPfiles_Feb2019")
    zps.write("/home/kadu/Dropbox/SPLUS/ifusci/tables/zps_Feb2019.fits",
              format="fits", overwrite=True)