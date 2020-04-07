# -*- coding: utf-8 -*-
""" 

Created on 06/04/20

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import yaml

import numpy as np
from astropy.table import Table, vstack

import context

def zp_tiles_table():
    with open("config_mode5.yaml") as f:
        config = yaml.load(f)
    zp_dir = os.path.join(config["main_dir"], "zeropoints")
    tiles_tables_dir = os.path.join(zp_dir, "tiles")
    mtable = []
    for tabfile in sorted(os.listdir(tiles_tables_dir)):
        tdata = Table.read(os.path.join(tiles_tables_dir, tabfile))
        mtable.append(tdata)
    mtable = vstack(mtable)
    for phot in ["FLUX_AUTO", "FLUX_APER"]:
        table = mtable[mtable["FLUXTYPE"]==phot]
        tiles = np.unique(table["TILE"])
        tout = []
        for tile in tiles:
            ttile = Table()
            ttile["TILE"] = [tile]
            for band in context.bands:
                idx = np.where((table["TILE"] == tile) &
                               (table["FILTER"]==band))[0]
                if len(idx) == 0:
                    ttile["ZP_{}".format(band)] = [np.nan]
                    ttile["ZPERR_{}".format(band)] = np.nan
                    continue
                t = table[idx]
                zp = np.mean(t["ZP"])
                zperr = np.sqrt(np.sum(t["ZPERR"]**2)) / len(idx)
                ttile["ZP_{}".format(band)] = [zp]
                ttile["ZPERR_{}".format(band)] = [zperr]
            tout.append(ttile)
        tout = vstack(tout)
        output = os.path.join(zp_dir, "zps_tiles-{}.fits".format(phot))
        tout.write(output, format="fits", overwrite=True)

def zp_singles_table():
    with open("config_mode5.yaml") as f:
        config = yaml.load(f)
    zp_dir = os.path.join(config["main_dir"], "zeropoints")
    single_tables_dir = os.path.join(zp_dir, "single")
    for phot in  ["FLUX_AUTO", "FLUX_APER"]:
        table = []
        for mode in ["mode0", "mode5"]:
            t = Table.read(os.path.join(single_tables_dir,
                                           "{}-{}.fits".format(mode, phot)))
            t.remove_column("GAIN")
            table.append(t)
        table = vstack(table)
        output = os.path.join(zp_dir, "zps_single-{}.fits".format(phot))
        table.write(output, format="fits", overwrite=True)




if __name__ == "__main__":
    # zp_tiles_table()
    zp_singles_table()