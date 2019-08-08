# -*- coding: utf-8 -*-
"""

Created on 22/08/2017

@Author: Carlos Eduardo Barbosa

Photometric callibration of standard star GD 50

"""
from __future__ import division, print_function

import os
import pickle

import numpy as np
from astropy.io import fits, ascii
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.table import Table, join, vstack
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from uncertainties import ufloat
from astropy.coordinates import SkyCoord

import context

ps = 0.5 * u.arcmin / u.pixel

class JypeCatPhot():
    """ Class to analyse and retrieve photometric results from the Jype
        pipeline. """
    def __init__(self, catfile, filter, airmass):
        """ Read data from pipeline and retrieve properties of GD 50 in a given
        catalog.

        Input parameters
        ----------------

        catfile : str
            Full path to the aperture photometry catalog from the Jype data
            products.
        """
        self.catfile = catfile
        self.logfile = catfile.replace(".apercorcat", "process.log")
        self.filter = filter
        self.airmass = airmass
        self.tbdata = Table.read(catfile, hdu=2)
        self.ra = self.tbdata["ALPHA_J2000"] * u.degree
        self.dec = self.tbdata["DELTA_J2000"] * u.degree
        self.flux = self.tbdata["FLUXAPERCOR"] * u.adu
        self.fluxerr = self.tbdata["FLUXERR_APERCOR"] * u.adu
        self.fwhm = self.tbdata["FWHM_WORLD"] * u.degree
        self.coords = SkyCoord(ra=self.ra, dec=self.dec)
        return

    def select_star(self, coords, full_output=False):
        """ Find star with nearest location in the catalog.

        Input Parameters
        ----------------

        coords : astropy.coords
            Coordinates of the star.

        full_output : bool
            Returns all the output or simply the catalog data.

        Output Parameters
        -----------------
        astropy.table
            Line of the catalog containing the star.

        d2d : float (optional)
            Projected distance of input and catalog coordinates.

        """
        idx, d2d, d3d = coords.match_to_catalog_sky(self.coords)
        if full_output:
            return self.tbdata[idx], d2d
        return Table(self.tbdata[idx])

    def select_psf_stars(self, magmin, magmax, coords, plot=False):
        """ Select stars in the catalog that have good flux and controled
        FWHM. """
        output = os.path.join(context.plots_dir,
                "stars_mag_fwhm_{}_X{:1.2f}.png".format(self.filter,
                                                        self.airmass))
        ###################################################################
        # # Selecting stars for analysis in the Jype catalog
        # median = np.median(imgdata)
        # skystd = 1.48 * np.median(np.abs(imgdata - median))
        # mag_5sig = mag(5 * skystd) + 25
        # magmax = gd50_jype["MAG_AUTO"] + 1.
        # magmin = mag_5sig - 11

        gd50 = self.select_star(coords) # reference star
        cat = self.tbdata[self.tbdata["MAG_BEST"] < magmax]
        cat = cat[cat["MAG_BEST"] > magmin]
        mask = ~sigma_clip(cat["FWHM_IMAGE"], sigma=2, iters=5).mask
        cat = cat[mask]
        if plot:
            ax = plt.subplot(111)
            ax.minorticks_on()
            ax.plot(self.tbdata["MAG_BEST"], self.tbdata["FWHM_IMAGE"], ".",
                     c="0.7")
            ax.plot(cat["MAG_BEST"], cat["FWHM_IMAGE"], ".",
                     c="C0")
            ax.plot(gd50["MAG_BEST"], gd50["FWHM_IMAGE"], "xr",
                     ms=10)
            plt.title("GD 50 filter: {}, X={:.2f}".format(self.filter,
                                                          float(self.airmass)))
            ax.axvline(x=magmin, c="C1", ls="--")
            ax.axvline(x=magmax, c="C1", ls="--")
            ax.set_xlim(np.percentile(self.tbdata["MAG_BEST"], 0.1),
                     np.percentile(self.tbdata["MAG_BEST"], 97))
            ax.set_ylim(np.percentile(self.tbdata["FWHM_IMAGE"], 2),
                     np.percentile(self.tbdata["FWHM_IMAGE"], 98))
            ax.set_xlabel("MAG_BEST")
            ax.set_ylabel("FWHM (pix)")
            plt.savefig(output, dpi=150)
            plt.clf()
        return cat

    def get_apercor(self):
        """ Obtain aperture correction from the logfile. """
        with open(self.logfile) as f:
            lines = f.readlines()
        self.apercor = {}
        for s in ["Median APERCOR:", "Mean APERCOR:", "Std APERCOR:"]:
            l = [_ for _ in lines if s in _][0]
            val = float(l.split(s)[1].strip())
            self.apercor[s[:-1]] = val
        return

def header_info_table(redo=False):
    """ Read file headers to determine the filter and airmass of the data. """

    filenames = sorted([_ for _ in context.gd50_dir if _.endswith(".fz")])
    output = os.path.join(context.tables_dir, "file_filter_airmass.txt")
    if os.path.exists(output) and not redo:
        return output
    filters, airmasses = [], []
    for fz in filenames:
        hdulist = fits.open(os.path.join(context.gd50_dir, fz))
        header = hdulist[1].header
        filters.append(header["FILTER"])
        airmasses.append(header["AIRMASS"])
    dat = Table([filenames, filters, airmasses],
                names=["filename", "filter", "airmass"])
    dat.write(output, format="ascii")

def get_GD50_jype(redo=False):
    """ Read catalogs produces by the Jype pipeline to get info about GD 50
    """
    intable = Table.read(header_info_table(redo=False), format="ascii")
    output = os.path.join(context.home_dir, "tables", "GD50_apphot_jype.fits")
    if os.path.exists(output) and not redo:
        return Table.read(output)
    filters = np.unique(intable["filter"])
    ###########################################################################
    # Properties of GD 50
    coords = SkyCoord("03h 48m 50.06s", "-00d 58' 30.4")
    tables = []
    for filter in filters:
        idx = np.where(intable["filter"] == filter)
        filenames = intable["filename"][idx]
        airmass = intable["airmass"][idx]
        for i, (fz, X) in enumerate(zip(filenames, airmass)):
            print("".join(["="] * 80))
            if filter == "U" and i == 5:
                continue
            print("Working on file {} ({} / {})".format(fz, i+1,
                                                        len(filenames)))
            catfile = os.path.join(context.gd50_dir, fz.replace(".fz", "files"),
                                   fz.replace(".fz", ".apercorcat"))
            ###################################################################
            # Getting data and header
            hdulist = fits.open(os.path.join(context.gd50_dir, fz))
            header = hdulist[1].header
            exptime = header["EXPTIME"]
            gain = header["GAIN"]
            ###################################################################
            # Obtaining aperture-corrected magnitude from catalog
            jcat = JypeCatPhot(catfile, filter, X)
            jcat.get_apercor()
            ###################################################################
            ramin = jcat.tbdata["ALPHA_J2000"].min()
            ramax = jcat.tbdata["ALPHA_J2000"].max()
            decmin = jcat.tbdata["DELTA_J2000"].min()
            decmax = jcat.tbdata["DELTA_J2000"].max()
            cmin = SkyCoord(ramin, decmin, unit=(u.degree, u.degree))
            cmax = SkyCoord(ramax, decmax, unit=(u.degree, u.degree))
            ###################################################################
            gd50 = jcat.select_star(coords)
            gd50["AIRMASS"] = X
            gd50["FILTER"] = filter
            gd50["EXPTIME"] = exptime
            gd50["FILENAME"] = fz
            tables.append(gd50)
    table = Table(vstack(tables))
    table["MAGAPERCOR"] = mag(table["FLUXAPERCOR"] / table["EXPTIME"])
    table.write(output, format="fits", overwrite=True)
    return table

def local_psf_phot(jcat, redo=False):
    """ Make local PSF photometry on stars using a reference catalog.

    Input Parameters
    ----------------
    jcat: astropy.Table
        Reference catalog produced by Jype pipeline

    redo: bool
        Redo photometry in case output file already exists.

    Output Parameters
    -----------------
    astropy.Table
        expanded catalog containing PSF magnitudes.

    """
    output = os.path.join(context.home_dir, "tables", "GD50_psfmag.fits")
    if os.path.exists(output) and not redo:
        return Table.read(output)
    amplitudes, alphas, betas = [], [], []
    for i, star in enumerate(jcat):
        print("Processing image {} ({} / {})".format(star["FILENAME"], i+1,
              len(jcat)))
        imgdata = fits.getdata(os.path.join(context.gd50_dir, star["FILENAME"]))
        rmax = 15 * star["FWHM_IMAGE"]
        xy = (star["X_IMAGE"], star["Y_IMAGE"])
        cutout = Cutout2D(imgdata, xy, [rmax * 2] * 2)
        x = np.arange(cutout.data.shape[0])
        xx, yy = np.meshgrid(x, x)
        x0, y0 = cutout.center_cutout
        ix0, iy0 = int(x0)-1, int(y0)-1
        amp0 = cutout.data[ix0, iy0]
        alpha = star["FWHM_IMAGE"] / 1.5
        beta = 2.5
        p0 = models.Moffat2D(amplitude=amp0, x_0=x0, y_0=y0, gamma=alpha,
             alpha=beta) + models.Polynomial2D(degree=0)
        fit = fitting.LevMarLSQFitter()
        p = fit(p0, xx, yy, cutout.data)
        if fit.fit_info["ierr"] not in [1, 2, 3, 4]:
            amplitudes.append(np.nan)
            alphas.append(np.nan)
            betas.append(np.nan)
        else:
            amplitudes.append(p.amplitude_0.value)
            alphas.append(p.gamma_0.value)
            betas.append(p.alpha_0.value)
    amplitudes = np.array(amplitudes)
    alphas = np.array(alphas)
    betas = np.array(betas)
    fwhm = alphas * 2 * np.sqrt(np.power(2, 1 / betas) - 1)
    tflux = amplitudes * np.pi * alphas * alphas / (betas - 1)
    mags = mag(tflux / jcat["EXPTIME"])
    jcat["PSFMAG"] = mags
    jcat.write(output, format="fits", overwrite=True)
    return

def GD50_model_fluxes(redo=False):
    """ Computes the flux of GD 50 by convolving spectrum with filters. """
    outtable = os.path.join(context.home_dir, "tables/gd50mags.pickle")
    if os.path.exists(outtable) and not redo:
        mags = pickle.load(open(outtable, "r"))
        return mags
    filters_dir = os.path.join(context.home_dir, "filters")
    filenames = sorted([_ for _ in os.listdir(filters_dir) if
                        _.endswith("txt")])
    specfile = os.path.join(context.home_dir, "spectrum/fgd50.dat")
    wave, flamb = np.loadtxt(specfile, usecols=(0,1), unpack=True)
    flamb = flamb * 1e-16 * u.erg / u.cm / u.cm / u.s / u.AA
    spec = interp1d(wave, flamb.value, kind="cubic", fill_value=0.,
                         bounds_error=False)
    gd50model = {}
    for fname in filenames:
        filter = fname.split(".")[0].replace("F0", "F").replace("SPLUS",
                                                            "").upper().strip()
        wave2, trans = np.loadtxt(os.path.join(filters_dir, fname)).T
        curve = interp1d(wave2, trans, kind="cubic", fill_value=0.,
                         bounds_error=False)
        term1 = lambda w: curve(w) * spec(w) * w
        term2 = lambda w: curve(w) / w
        I1 = quad(term1, wave2.min(), wave2.max())
        I2 = quad(term2, wave2.min(), wave2.max())
        cAAs = c.to(u.AA / u.s).value
        fnu = I1[0] / I2[0] / cAAs
        mab = -2.5*np.log10(fnu) - 48.6
        gd50model[filter] = mab
    pickle.dump(gd50model, open(outtable, "wb"))
    return gd50model

def calc_zp(cat, model):
    """ Calculates the photometric zero point and extinction coefficient
    for GD 50 data.

    """
    def residue(p, M, m, airmass, error):
        return (M - m -(p[0] - p[1] * airmass)) / error
    p0 = np.array([20, 0.5])
    fdata = ascii.read(os.path.join(context.tables_dir, "filter_lam.txt"))
    fdata = fdata[np.argsort(fdata["lam(AA)"])]
    Nsim = 200
    magkeys = ["PSFMAG", "MAGAPERCOR"]
    magids = ["New calibration", "Jype pipeline"]
    filenames = ["newcalib", "jype"]
    for j, mkey in enumerate(magkeys):
        for filter in fdata["Filter"]:
            idx = np.argwhere(cat["FILTER"] == filter)
            subdata = cat[idx]
            subdata_nofilter = np.copy(subdata)
            filtered_data = sigma_clip(subdata[mkey], sigma=2.5)
            subdata = subdata[~filtered_data.mask]
            M = model[filter]
            m = np.array(subdata[mkey]).T
            merr = np.ones_like(m) * 0.05
            X = np.array(subdata["AIRMASS"]).T
            fit = least_squares(residue, p0, args=(M, m, X, merr),
                                loss="cauchy")
            p = fit["x"]
            x = np.linspace(np.floor(X.min()), X.max(), 100)
            plt.clf()
            ax = plt.subplot(111)
            ax.minorticks_on()
            ax.set_xlabel("$X$")
            ax.set_ylabel("$m_0 - \kappa \cdot X$", size=12)
            ax.plot(subdata_nofilter["AIRMASS"],
                     M - subdata_nofilter[mkey],
                     "o", label="GD 50")
            plt.plot(x, p[0] - p[1] * x, ls="-", c="C1", label="Best fit")
            plt.title("{}, band : {}".format(magids[j], filter))
            ax.legend(loc=1)
            # Simulation for estimating the errors in the parameters
            # Generating simulated data:
            mockdata = np.zeros((Nsim, len(m)))
            for i in np.arange(len(m)):
                mockdata[:,i] = np.random.normal(m[i], merr[i], Nsim)
            fsim = np.zeros((Nsim, 2))
            for N in np.arange(Nsim):
                fsim[N] = least_squares(residue, p0, args=(M, mockdata[N], X, \
                                                        merr), loss="cauchy")["x"]
                plt.plot(x, fsim[N][0] - fsim[N][1] * x, ls="-", c="C1",
                         alpha=0.1, linewidth=0.5)
            err = np.std(fsim, axis=0)
            m0 = ufloat(p[0], err[0])
            kappa = ufloat(p[1], err[1])
            ax.text(0.05, 0.15, "$m_0={:L}$".format(m0), transform=ax.transAxes,
                    fontsize=15)
            ax.text(0.05, 0.08, "$\kappa={:L}$".format(kappa),
                    transform=ax.transAxes, fontsize=15)
            pdir = os.path.join(context.plots_dir, "zps")
            if not os.path.exists(pdir):
                os.mkdir(pdir)
            plt.savefig(os.path.join(pdir, "{}_{}.png".format(filenames[j],
                                                              filter)))

def mag(f):
    return -2.5 * np.log10(f)

if __name__ == "__main__":
    intable = header_info_table(redo=False)
    model = GD50_model_fluxes(redo=False)
    jypephot = get_GD50_jype(redo=True)

    # psfphot = local_psf_phot(jypephot, redo=False)
    # calc_zp(psfphot, model)
