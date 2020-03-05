# -*- coding: utf-8 -*-
""" 

Created on 07/10/19

Author : Carlos Eduardo Barbosa

Calculate pivot wavelength of filters.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
from scipy.integrate import simps

def calc_wpiv(wave, T):
    num = simps(T * wave, wave)
    den = simps(T / wave, wave)
    return np.sqrt(num / den) * u.AA


def splus_wave_pivot(bands, dr="dr1"):
    """ Calculates the pivot wavelength for S-PLUS bands. """
    filters_dir = os.path.join(os.getcwd(),
                               "tables/filter_curves/{}".format(dr))
    ftable = ascii.read(os.path.join(os.getcwd(), "tables",
                                     "filter_lam_filename.txt"))
    wave_piv = {}
    for f in ftable:
        if f["filter"] not in bands:
            continue
        fname = os.path.join(filters_dir, f["filename"])
        fdata = np.loadtxt(fname)
        fdata = np.vstack(((fdata[0,0]-1, 0), fdata, (fdata[-1,0]+1, 0)))
        wave = fdata[:,0]
        T = np.clip(fdata[:,1], 0., 1.)
        num = simps(T * wave, wave)
        den = simps(T / wave, wave)
        wave_piv[f["filter"]] = np.sqrt(num / den) * u.AA
    return wave_piv

def panstarrs_wave_pivot():
    filters_dir = os.path.join(os.getcwd(), "tables/panstarrs")
    wave_piv = {}
    for table in os.listdir(filters_dir):
        band = table.split(".")[1]
        wave, T = np.loadtxt(os.path.join(filters_dir, table)).T
        wave_piv[band] = calc_wpiv(wave, T) * u.AA
    return wave_piv

def DES_wave_pivot():
    filenames = os.path.join(os.getcwd(),
                             "tables/DES/STD_BANDPASSES_DR1.dat.txt")
    data = Table.read(filenames, format="ascii")
    wpiv = {}
    for band in data.colnames[1:-1]:
        wpiv[band] = calc_wpiv(data["LAMBDA"], data[band])
    return wpiv

if __name__ == "__main__":
    bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I',
             'F861', 'Z']
    wpiv_SPLUS = splus_wave_pivot(bands)
    wpiv_panstarrs = panstarrs_wave_pivot()
    wpiv_des = DES_wave_pivot()