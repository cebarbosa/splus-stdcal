# -*- coding: utf-8 -*-
"""

Created on 06/10/2017

@Author: Carlos Eduardo Barbosa

Program to handle convolutions of spectra with filter system.

"""

from __future__ import division, print_function

import os

import astropy.units as u
from astropy import constants
import numpy as np
from scipy.interpolate import interp1d

import context

def convolve2splus(wave, flux):
    """ Calculates the AB magnitudes of a flux-calibrated spectra according
    the the SPLUS filter system.

    Input Parameters
    ----------------
    wave : astropy.Quantity
        Array containing the wavelength of the spectrum with units.

    flux : astropy.Quantity
        Array containining the flux os the spectrum with units.

    """
    filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "tables/filter_curves-master")
    filenames = sorted([_ for _ in os.listdir(filters_dir) if
                        _.endswith("dat")])
    cAAs = constants.c.to(u.AA / u.s).value
    mags = {}
    for fname in filenames:
        filtername = fname.split(".")[0].replace("F0", "F").replace("SDSS",
                           "").replace("JAVA", "").upper().strip()
        wave2, trans = np.loadtxt(os.path.join(filters_dir, fname)).T
        curve = interp1d(wave2, trans, kind="linear", fill_value=0.,
                         bounds_error=False)
        trans = curve(wave)
        term1 = np.trapz(trans * flux * wave, wave)
        term2 = np.trapz(trans / wave, wave)
        print(filtername, np.average(wave, weights=trans))
        fnu = term1 / term2 / cAAs
        mab = -2.5 * np.log10(fnu.value) - 48.6
        mags[filtername] = mab
    return mags

def example_gd50():
    """ Computes the flux of GD 50 by convolving spectrum with filters from
    SPLUS. """
    specfile = os.path.join(context.home_dir, "spectrum/fgd50.dat")
    wave, flamb = np.loadtxt(specfile, usecols=(0, 1), unpack=True)
    wave = wave * u.AA
    flamb = flamb * 1e-16 * u.erg / u.cm / u.cm / u.s / u.AA
    splusmags = convolve2splus(wave, flamb)
    print(splusmags)
    return

if __name__ == "__main__":
    example_gd50()