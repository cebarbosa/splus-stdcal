# -*- coding: utf-8 -*-
""" 

Created on 25/02/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os

from astropy.table import Table, vstack
from tqdm import tqdm

import context

def join_tables(redo=False):
    """ Join standard star catalogs into one table. """
    print("Searching files...")
    output = os.path.join(context.tables_dir, "extmoni_phot.fits")
    if os.path.exists(output) and not redo:
        return output
    datadir = os.path.join(context.data_dir, "extmoni")
    dates = sorted([_ for _ in os.listdir(datadir) if os.path.isdir(
                   os.path.join(datadir, _))])
    fields = ["DATE", "IMAGE", "STAR", "FILTER", "AIRMASS", "EXPTIME", "D2D",
              "FLAGS"]
    fluxes = ["FLUX_AUTO", "FLUX_ISO", "FLUXAPERCOR"]
    errors = ["FLUXERR_AUTO", "FLUXERR_ISO","FLUXERR_APERCOR"]
    list_of_tables = []
    for date in dates:
        datedir = os.path.join(datadir, date)
        tables = [_ for _ in os.listdir(datedir) if _.endswith(".fits")]
        for table in tables:
            list_of_tables.append(os.path.join(datedir, table))
    outtable = []
    for table in tqdm(list_of_tables):
        data = Table.read(table)
        outtable.append(data[fields + fluxes + errors])
    print("Joining tables and saving the results...")
    outtable = vstack(outtable)
    # Fix the name of the stars
    stars = [_.lower().replace(" ", "") for _ in outtable["STAR"]]
    outtable["STAR"] = stars
    outtable.write(output, format="fits", overwrite=True)
    return output

if __name__ == "__main__":
    phottable = join_tables(redo=True)