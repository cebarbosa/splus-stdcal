# -*- coding: utf-8 -*-
"""

Created on 06/10/2017

@Author: Carlos Eduardo Barbosa

Perform photometric calibration of GD 50 field using spectroscopic data from
Stripe 82.

"""

from __future__ import division, print_function

import os

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, hstack, vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from uncertainties import ufloat
import speclite.filters

import context
from gd50cal import header_info_table


def convolve_strip82_to_photsys(table, outputs, photsystem="splus",
                              redo=False, ):
    """ Convolve spectra with filter to obtain magnitudes AB in the SPLUS
    system.

    Input Parameters
    ----------------
    table : astropy.Table
        SDSS table obtained with the download from stripe 82

    outputs : list containing strings
        List containing the output files for 1) results using extrapolated
        spectrum and 2) results for model only

    redo : bool
        Redo work in case output already exists.

    """
    if os.path.exists(outputs[0]) and not redo:
        print("Warning: Using magnitudes from existing tables.")
        t1 = Table(fits.getdata(outputs[0], 1))
        t2 = Table(fits.getdata(outputs[1], 1))
        return t1, t2
    print("Performing calculations of magnitudes of Stripe 82 stars...")
    # Registering SPLUS filter on speclite
    if photsystem == "splus":
        photsys = load_splus_filters()
    elif photsystem == "sdss":
        photsys = speclite.filters.load_filters("sdss2010-*")
    # plt.show()
    # Load stellar library to extend spectrum of stars
    ngsl = load_ngsl()
    # Setting up input spectra
    specs_dir = os.path.join(context.data_dir, "stripe82")
    # Preparing output
    mags1, mags2 = [], []
    nanline = Table(data=np.zeros(len(photsys.names)) * np.nan,
                    names=photsys.names)
    for i, spec in enumerate(table):
        sname = "spec-{}-{}-{:04d}.fits".format(spec["plate"], spec["mjd"],
                                            spec["fiberid"])
        print("Working on spectrum {} ({}/{})".format(sname, i+1, len(table)))
        filename = os.path.join(specs_dir, sname)
        if not os.path.exists(filename):
            print("Filename does not exist: {}".format(sname))
            mags1.append(nanline)
            mags2.append(nanline)
            continue
        print("Using file {}".format(sname))
        hdulist = fits.open(filename)
        tbdata = hdulist[1].data
        wave = np.power(10, tbdata["loglam"])
        flux = tbdata["flux"]
        sigma = np.sqrt(1 / tbdata["ivar"])
        mask = np.where(np.logical_or(tbdata["and_mask"] <= 1,
                                      tbdata["and_mask"] == 4))[0]
        if len(mask) == 0:
            print("Ignoring spectrum for lack of data: {}".format(sname))
            mags1.append(nanline)
            mags2.append(nanline)
            continue
        wave = wave[mask].astype(np.float)
        flux = flux[mask].astype(np.float)
        sigma = sigma[mask].astype(np.float)
        wave = wave * u.AA
        flux = flux * 1e-17 * u.erg / u.cm / \
               u.cm / u.s / u.AA
        sigma = sigma * 1e-17 * u.erg / u.cm / \
               u.cm / u.s / u.AA
        mwave, mflux = extrapolate_spectrum(wave, flux, sigma, ngsl)
        #######################################################################
        # Making hybrid spectrum
        idx1 = np.where(mwave < wave[0])[0]
        idx2 = np.where(mwave > wave[-1])[0]
        funit = flux.unit
        flux = np.hstack((mflux[idx1].value, flux.value,
                          mflux[idx2].value)) * funit
        wave = np.hstack((mwave[idx1].value, wave.value,
                          mwave[idx2].value)) * u.AA
        #######################################################################
        m1 = photsys.get_ab_magnitudes(flux, wave, mask_invalid=True)
        m2 = photsys.get_ab_magnitudes(mflux, mwave, mask_invalid=True)
        sn1 = der_snr_bands(wave, flux)
        sn2 = der_snr_bands(mwave, mflux)
        mags1.append(hstack([m1, sn1]))
        mags2.append(hstack([m2, sn1]))
        # if True:
        #     tabpath = os.path.join(paths.tables_dir, "filter_lam_filename.txt")
        #     filtertab = ascii.read(tabpath)
        #     m = np.array([float(m2["splus-{}".format(_)]) for _ in filtertab["filter"]])
        #     fnu = np.power(10., -0.4 * (m + 48.6)) * u.erg / u.s / u.cm / u.cm / u.Hz
        #     lam = np.array([_ for _ in filtertab["lam"]]) * u.AA
        #     flam = fnu / lam / lam * constants.c
        #     flam = flam.to(funit).value
        #     ax = plt.subplot(111)
        #     ax.plot(wave, flux, "-", label="Data", ms=0.5)
        #     ax.plot(mwave, mflux, label="Model")
        #     plt.plot(lam, flam, "o", label="m(AB)")
        #     ax.legend()
        #     plt.show()
    tabs = []
    for m, out in zip([mags1, mags2], outputs):
        results = vstack(m)
        # for name in results.colnames:
        #     results.rename_column(name, name.split("-")[-1])
        results.write(out, format="fits", overwrite=True)
        tabs.append(results)
    return tabs[0], tabs[1]

def load_splus_filters():
    """ Use speclite to load SPLUS filters for convolution. """
    filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "tables/filter_curves-master")
    ftable = ascii.read(os.path.join(context.tables_dir,
                                     "filter_lam_filename.txt"))
    filternames = []
    for f in ftable:
        fname = os.path.join(filters_dir, f["filename"])
        fdata = np.loadtxt(fname)
        fdata = np.vstack(((fdata[0,0]-1, 0), fdata, (fdata[-1,0]+1, 0)))
        w = fdata[:,0] * u.AA
        response = np.clip(fdata[:,1], 0., 1.)
        speclite.filters.FilterResponse(wavelength=w,
                  response=response, meta=dict(group_name="splus",
                  band_name=f["filter"]))
        filternames.append("splus-{}".format(f["filter"]))
    splus = speclite.filters.load_filters(*filternames)
    return splus

def load_ngsl():
    """ Load spectrum from NGSL library"""
    ngslpath = os.path.join(context.home_dir, "stis_ngsl_v2")
    specs = [_ for _ in os.listdir(ngslpath) if _.endswith(".fits")]
    ngsl = []
    for spec in specs:
        ngsl.append(Table(fits.getdata(os.path.join(ngslpath, spec))))
    return ngsl

def extrapolate_spectrum(wave, flux, sigma, lib):
    """ Extrapolate spectrum based on stellar library model. """
    sols = np.zeros((len(lib), len(wave)))
    costs = np.zeros(len(lib))
    scales = np.zeros_like(costs)
    for i, data in enumerate(lib):
        ws = data["WAVELENGTH"]
        fs = data["FLUX"]
        spec = interp1d(ws, fs, kind="linear", fill_value=0, bounds_error=False)
        star = spec(wave)
        idx = np.where(np.logical_and(wave.to("AA").value >= 5010,
                                      wave.to("AA").value <= 5060))
        scale0 = 1e-6
        diff = lambda A: (np.array(flux) - A * star) / np.array(sigma)
        sol = least_squares(diff, scale0)
        sols[i] = sol["x"] * star
        scales[i] = sol["x"]
        costs[i] = np.sqrt(np.sum(diff(sol["x"])**2)) / len(wave)
    idx = np.argmin(costs)
    best = lib[idx]
    widx = np.where(best["WAVELENGTH"] > 9200)
    z = np.polyfit(best["WAVELENGTH"][widx], scales[idx] * best["FLUX"][widx], 1)
    p = np.poly1d(z)
    dlam = np.diff(best["WAVELENGTH"])[-1]
    exlam = np.arange(best["WAVELENGTH"][-1] + dlam, 10600, dlam)
    mwave = np.hstack((best["WAVELENGTH"], exlam)) * u.AA
    mflux = np.hstack((scales[idx] * best["FLUX"], p(exlam))) * u.erg / u.cm / \
               u.cm / u.s / u.AA
    return mwave, mflux

def der_snr_bands(wave, flux):
    """ Use DER_SNR algorithm to calculate S/N in SPLUS bands."""
    noise_spec = np.abs(2 * np.roll(flux, -2) - np.roll(flux, -4) - flux)
    filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "tables/filter_curves-master")
    filenames = sorted([_ for _ in os.listdir(filters_dir) if
                        _.endswith("dat")])
    cols = []
    for fname in filenames:
        filtername = fname.split(".")[0].replace("F0", "F").replace("SDSS",
                           "").replace("JAVA", "").upper().strip()
        filtername = "SNR-splus-{}".format(filtername)
        wave2, trans = np.loadtxt(os.path.join(filters_dir, fname)).T
        wave2 = wave2 * u.AA
        idx = np.where(np.logical_and(wave >= wave2.min(),
                                      wave <= wave2.max()))[0]
        sn = np.median(flux[idx]) / (0.6052697 * np.median(noise_spec[idx]))
        cols.append(Table(data=np.atleast_2d(sn), names=[filtername,]))
    return hstack(cols)

def compare_magnitudes_sdss_catalog():
    """ Compare the model magnitudes of the models with those from the
    catalogs. """
    sdsscat = os.path.join(context.tables_dir, "stars_around_gd50_kadu.fit")
    modelcat = os.path.join(context.tables_dir, "stripe82_splus_aroundgd50.fits")
    sdss = fits.getdata(sdsscat, 1)
    model = fits.getdata(modelcat, 1)
    filters = ("U", "G", "R", "I", "Z")
    for filter in filters:
        msplus = model[filter]
        msdss = sdss["petromag_{}".format(filter.lower())]
        plt.plot(msdss, msplus - msdss, "o")
        plt.title("Band: {}".format(filter))
        plt.axhline(y=0, ls="--", c="k")
        print(np.nanmedian(msplus - msdss))
        plt.axhline(y=np.nanmedian(msplus - msdss), c="r")
        plt.ylim(plt.ylim()[::-1])
        plt.show()

def compare_magnitudes_marcus():
    """ Make comparison of magnitudes in my convolution with models from
    Marcus"""
    mvdcat = os.path.join(context.tables_dir,
                          "sample_stars_Cadu_SPLUS_synthetic_mags")
    mvd = ascii.read(mvdcat)
    sdsscat = os.path.join(context.tables_dir, "stars_around_gd50_kadu.fit")
    modelcat = os.path.join(context.tables_dir, "stripe82_splus_aroundgd50.fits")
    sdss = Table(fits.getdata(sdsscat, 1))
    ceb = Table(fits.getdata(modelcat, 1))
    aid = ["{}.{}.{:04d}".format(a,b,c) for a,b,c in
           sdss[["plate", "mjd", "fiberid"]]]
    ceb["aid"] = aid
    idx = []
    for aid in mvd["aid"]:
        idx.append(np.argwhere(ceb["aid"]==aid)[0][0])
    ceb = ceb[idx]
    sdss = sdss[idx]
    filters = ascii.read(os.path.join(context.tables_dir, "filter_lam.txt"))
    for filter in ["U", "G", "R", "I", "Z"]:
        filter2 = "petromag_{}".format(filter.lower())
        plt.plot(sdss[filter2], mvd[filter]- sdss[filter2], "o",
                 label="Simulation")
        plt.plot(sdss[filter2], ceb[filter] - sdss[filter2], "o")
        plt.show()

def compare_mags_obs_synth():
    obscat = os.path.join(context.tables_dir, "stripe82_splus_aroundgd50.fits")
    modcat = obscat.replace(".fits", "_model.fits")
    obs = Table(fits.getdata(obscat, 1))
    mod = Table(fits.getdata(modcat, 1))
    for band in obs.colnames:
        plt.plot(obs[band], mod[band], "o")
        plt.title(band)
        plt.show()

def make_photcal_tables(table, redo=False, photname="spec"):
    """ Find stars in the Jype catalogs and retrieve their aperture magnitudes.
    """
    catalog = SkyCoord(ra=table["ra"] * u.degree,
                       dec=table["dec"]*u.degree)
    jype_cats = Table.read(header_info_table(redo=False), format="ascii")
    filters = np.unique(jype_cats["filter"])
    outdir = os.path.join(context.tables_dir, "photcal_stripe82")
    for j, filter in enumerate(filters):
        idx = np.where(jype_cats["filter"] == filter)
        filenames = jype_cats["filename"][idx]
        airmass = jype_cats["airmass"][idx]
        outfile = os.path.join(outdir, "{}_{}.fits".format(photname, filter))
        if os.path.exists(outfile) and not redo:
            continue
        tables = []
        for i, (fz, X) in enumerate(zip(filenames, airmass)):
            print("".join(["="] * 80))
            print("Working on catalog {} (File: {} / {}, Filter: {} / {})".format(
                  fz, i+1, len(filenames), j+1, len(filters)))
            catfile = os.path.join(context.gd50_dir, fz.replace(".fz", "files"),
                                   fz.replace(".fz", ".apercorcat"))
            # Getting data and header
            hdulist = fits.open(os.path.join(context.gd50_dir, fz))
            header = hdulist[1].header
            exptime = header["EXPTIME"]
            gain = header["GAIN"]
            ###################################################################
            catdata = Table.read(catfile, hdu=2)
            raj = catdata["ALPHA_J2000"] * u.degree
            decj = catdata["DELTA_J2000"] * u.degree
            cj = SkyCoord(ra=raj, dec=decj)
            idx, d2d, d3d = catalog.match_to_catalog_sky(cj)
            matchcat = catdata[idx][["FLUXAPERCOR", "FLUXERR_APERCOR"]]
            matchcat["MAGAPCOR"] = -2.5 * np.log10(matchcat["FLUXAPERCOR"] /  \
                                                   exptime)
            matchcat["MAGERRAPCOR"] = np.abs(2.5 / np.log(10.) * \
                                      matchcat["FLUXERR_APERCOR"] / \
                                      matchcat["FLUXAPERCOR"])
            newcat = hstack([table, matchcat])
            idxdist = np.argwhere(d2d < 1.5 * u.arcsec)
            newcat = newcat[idxdist]
            newcat["AIRMASS"] = np.ones(len(idxdist)) * X
            tables.append(newcat)
        results = vstack(tables)
        print(results)
        results.write(outfile, format="fits", overwrite=True)

def call_photcal(photname="spec", sn_min=0):
    """ Make photometric calibration for a given type of photometry. """
    tabdir = os.path.join(context.tables_dir, "photcal_stripe82")
    tablenames = [_ for _ in os.listdir(tabdir) if _.startswith(photname)]
    # colors = {"U" : ("U", "G"), "G" : ("G", "R"), "R" : ("G", "R"),
    #           "I" : ("R", "I"), "Z" : ("I", "Z"), "F378" : ("F378", "F395"),
    #           "F395" : ("F395", "F410"), "F410" : ("F410", "F430"),
    #           "F430": ("F430", "F515"), "F515": ("F430", "F515"),
    #           "F660" : ("F515", "F660"), "F861" : ("F660", "F861")}
    colors = {"U" : ("U", "G"), "G" : ("G", "R"), "R" : ("G", "R"),
              "I" : ("R", "I"), "Z" : ("R", "I"), "F378" : ("G", "R"),
              "F395" : ("G", "R"), "F410" : ("G", "R"),
              "F430": ("G", "R"), "F515": ("G", "R"),
              "F660" : ("R", "I"), "F861" : ("R", "I")}
    for table in tablenames:
        band = table.split(".")[0].split("_")[1]
        print(band)
        data = Table(fits.getdata(os.path.join(tabdir, table)))
        c1 = "splus-{}".format(colors[band][0])
        c2 = "splus-{}".format(colors[band][1])
        # Selecting good S/N data
        sn1 = data["SNR-{}".format(c1)]
        sn2 = data["SNR-{}".format(c2)]
        sn = np.minimum(sn1, sn2)
        good = np.where(sn > sn_min)[0]
        cat = []
        for field in ["splus-{}".format(band), "MAGAPCOR", "AIRMASS", "MAGERRAPCOR"]:
            cat.append(np.array(data[field][good], dtype=np.float))
        c = np.array(data[c1][good] - data[c2][good], dtype=np.float)
        cat.append(c)
        cat = np.array(cat).T
        # Filtering data to remove nans
        cat = cat[~np.isnan(cat).any(axis=1)]
        # # Getting only brightest stars
        # cat = cat[np.argsort(cat[:,0])][:5*10]
        photcal(cat)

def photcal(data):
    """ Calculates the photometric zero point and extinction coefficient.

    """
    def residue(p, M, m, airmass, c, error):
        return (M - m -(p[0] - p[1] * airmass + c * p[2]))
    M, m, X, merr, c = data.T
    p0 = np.array([20, 0.5, 0.])
    Nsim = 200
    fit = least_squares(residue, p0, args=(M, m, X, c, merr), loss="soft_l1",
                        f_scale=0.5)
    p = fit["x"]
    print(p)
    try:
        fig = plt.figure()
        ax = plt.subplot(111)
        # ax = Axes3D(fig)
        ax.scatter(X, M - m - (p[0] - p[1] * X + p[2] * c))
        plt.show()
    except:
        pass
    return
    ax.set_xlabel("$X$")
    ax.set_ylabel("$m_0 - \kappa \cdot X$", size=12)
    ax.plot(M, M - m - (p[0] - p[1] * x + p[2] * c), "o", label="GD 50")
    plt.show()
    plt.plot(X, M - m - (p[0] - p[1] * x + p[2] * c), ls="-", c="C1", label="Best fit")
    plt.legend()
    plt.show()
    # Simulation for estimating the errors in the parameters
    # Generating simulated data:
    mockdata = np.zeros((Nsim, len(data)))
    for i in np.arange(len(data)):
        mockdata[:,i] = np.random.normal(m[i], merr[i], Nsim)
    fsim = np.zeros((Nsim, 2))
    for N in np.arange(Nsim):
        fsim[N] = least_squares(residue, p0, args=(M, mockdata[N], X, \
                                                c, merr), loss="cauchy")["x"]
        plt.plot(x, fsim[N][0] - fsim[N][1] * x, ls="-", c="C1",
                 alpha=0.1, linewidth=0.5)
    err = np.std(fsim, axis=0)
    m0 = ufloat(p[0], err[0])
    kappa = ufloat(p[1], err[1])
    ax.text(0.05, 0.15, "$m_0={:L}$".format(m0), transform=ax.transAxes,
            fontsize=15)
    ax.text(0.05, 0.08, "$\kappa={:L}$".format(kappa),
            transform=ax.transAxes, fontsize=15)
    plt.show()
    # pdir = os.path.join(paths.plots_dir, "zps")
    # if not os.path.exists(pdir):
    #     os.mkdir(pdir)
    # plt.savefig(os.path.join(pdir, "{}_{}.png".format(filenames[j],
    #                                                   filter)))


if __name__ == "__main__":
    tablename = os.path.join(context.tables_dir, "stars_around_gd50_kadu.fit")
    sdsstable = Table.read(tablename)
    output1 = os.path.join(context.tables_dir, "stripe82_splus_aroundgd50.fits")
    output2 = output1.replace(".fits", "_model.fits")
    outputs = [output1, output2]
    specmags, modelmags = convolve_strip82_to_photsys(sdsstable, outputs,
                                                      redo=False)
    table1 = hstack([sdsstable, specmags])
    table2 = hstack([sdsstable, modelmags])
    # compare_magnitudes_sdss_catalog()
    # compare_magnitudes_marcus()
    # compare_mags_obs_synth()
    make_photcal_tables(table1, redo=False, photname="spec")
    # make_photcal_tables(table2, redo=False, photname="mod")
    call_photcal(photname="spec")