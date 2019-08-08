# -*- coding: utf-8 -*-
""" 

Created on 11/04/18

Author : Carlos Eduardo Barbosa

Script to read zero points for all calibrated tiles and make a table.

"""
from __future__ import print_function, division

import os

from astropy.table import Table, vstack

import context

def get_zeropoints():
    """ Read data from files and make a single table. """
    data_dir = os.path.join(context.data_dir, "tiles")
    zps = []
    for tile in os.listdir(data_dir):
        fname = os.path.join(data_dir, tile, "zeropoints.fits")
        if not os.path.exists(fname):
            continue
        zps.append(Table.read(fname, format="fits"))
    zps = vstack(zps)
    output = os.path.join(context.tables_dir, "tiles_zeropoints.fits")
    zps.write(output, format="fits", overwrite=True)

if __name__ == "__main__":
    get_zeropoints()