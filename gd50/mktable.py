# -*- coding: utf-8 -*-
"""

Created on 26/09/2017

@Author: Carlos Eduardo Barbosa

Produces table of GD 50 data used for calibration.

"""
from __future__ import division, print_function

from datetime import datetime

from astropy.io import fits, ascii
from astropy.table import Table

from context import *

def make_table():
    """ Produces table with information about data. """
    os.chdir(data_dir)
    fzfiles = sorted([_ for _ in os.listdir(data_dir) if _.endswith(".fz")])
    dates, times, exptimes, filters, airmasses = [], [], [], [], []
    for fz in fzfiles:
        header = fits.getheader(fz, 1)
        date = datetime.strptime(header["DATE"], "%Y-%m-%dT%H:%M:%S.%f")
        dates.append(date.date())
        times.append(str(date.time())[:-4])
        exptimes.append(header["EXPTIME"])
        filters.append(header["FILTER"])
        airmasses.append("{:.2f}".format(header["AIRMASS"]))
    fnames = [_.replace("_", "\_") for _ in fzfiles]
    table = Table([fnames, dates, times, exptimes, filters, airmasses],
                  names=["file", "date", "time", "exptime", "filter",
                         "airmass"])
    ascii.write(table, os.path.join(tables_dir, "info.tex"), format="latex",
                overwrite=True)


if __name__ == "__main__":
    make_table()

