# -*- coding: utf-8 -*-
""" 

Created on 07/11/18

Author : Carlos Eduardo Barbosa

Produces a table with the nights with reliable photometric calibration

"""
from __future__ import print_function, division

import os

from astropy.table import Table

import context

if __name__ == "__main__":
    # The reference folder where the plots are stored
    good_dir = os.path.join(context.plots_dir, "good_calib")
    nights = sorted(list(set([_.split("_")[0] for _ in os.listdir(good_dir)])))
    table = Table([nights], names=["night"])
    table.write("tables/photcal_good_nights.fits", overwrite=True)
