# -*- coding: utf-8 -*-
""" 

Created on 15/03/18

Author : Carlos Eduardo Barbosa

Make the calibration of tiles based on single catalogs.

"""

from __future__ import print_function, division

import os
import sys
import yaml
from subprocess import call
from functools import partial

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from multiprocessing import Pool

import context

def run_sextractor_tile(fitsfile, outcat):
    """ Runs SExtractor on stamps of standard stars. """
    # Load SExtractor parameters
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                              "config")
    sex_config_file = os.path.join(config_dir, "stdcal.sex")
    sex_params_file = os.path.join(config_dir, "stdcal.param")
    filter_file = os.path.join(config_dir, "gauss_3.0_5x5.conv")
    starnnw_file = os.path.join(config_dir, "default.nnw")
    call(["sextractor", fitsfile, "-c", sex_config_file,
          "-PARAMETERS_NAME", sex_params_file,
          "-FILTER_NAME", filter_file, "-STARNNW_NAME", starnnw_file,
          "-CATALOG_NAME", outcat])
    return

def make_reference_catalog(cats, fluxkey, errkey):
    """ Join catalogs of single exposures and make reference catalog for
    photometry of tiles"""
    if len(cats) == 1:
        return cats[0]
    coords = [SkyCoord(ra=cat["ALPHA_J2000"],
                       dec=cat["DELTA_J2000"]) for cat in cats]
    ############################################################################
    # Aligning catalogs
    newcats = []
    for i, (c, cat) in enumerate(zip(coords, cats)):
        if i == 0:
            newcats.append(cat)
            continue
        c0 = SkyCoord(ra=newcats[0]["ALPHA_J2000"],
                      dec=newcats[0]["DELTA_J2000"])
        ci = SkyCoord(ra=cat["ALPHA_J2000"], dec=cat["DELTA_J2000"])
        idx, d2d, d3d = c0.match_to_catalog_sky(ci)
        goodidx = np.where(d2d.to("arcsec").value < 1)[0]
        newcat = cat[idx][goodidx]
        newcats.append(newcat)
        for j in range(len(newcats) -1):
            newcats[j] = newcats[j][goodidx]
    # Calibrating in flux
    flux = np.dstack([dat[fluxkey].quantity.value for dat in newcats])
    errors = np.dstack([dat[errkey].quantity.value for dat in newcats])
    weights = np.power(errors, -2)
    meanf = np.average(flux, weights=weights, axis=-1)
    meanferr = np.sqrt(np.sum(np.power(weights * errors, 2), axis=-1)) / \
                       np.sum(weights, axis=-1)
    newcat[fluxkey] = meanf
    newcat[errkey] = meanferr
    return newcat

def calib_tile(config, bands, zptables, tile, verbose=False, redo=False):
    """ Performs the calibration of tiles using photometric single exposures """
    output = os.path.join(config["out_dir"], "{}-{}.fits".format(tile,
                                                                config["name"]))
    if os.path.exists(output) and not redo:
        return
    errkeys = {"FLUX_AUTO": "FLUXERR_AUTO", "FLUX_APER": "FLUXERR_APER"}
    cat_dir = os.path.join(config["cats_dir"], tile)
    tile_dir = os.path.join(config["tiles_dir"], tile)
    results = []
    for band in bands:
        print(tile, band)
        imgname = "{}_{}_swp.fits".format(tile, band)
        imgpath = os.path.join(tile_dir, band, imgname)
        fzfile = imgpath.replace(".fits", ".fz")
        tmp_fits = os.path.join(config["tmp_dir"], imgname)
        catfile = os.path.join(cat_dir, imgname.replace(".fits", ".cat"))
        if not os.path.exists(catfile) or config["sex_redo"]:
            if not os.path.exists(fzfile):
                continue
            if not os.path.exists(cat_dir):
                os.mkdir(cat_dir)
            if os.path.exists(imgpath):
                if verbose:
                    print("Running SExtractor...")
                run_sextractor_tile(imgpath, catfile)
            else:
                if not os.path.exists(config["tmp_dir"]):
                    os.mkdir(config["tmp_dir"])
                if verbose:
                    print("Unpacking fz file...")
                call(["funpack", "-O", tmp_fits, fzfile])
                if verbose:
                    print("Running SExtractor...")
                run_sextractor_tile(tmp_fits, catfile)
                os.remove(tmp_fits)
        tilecat = Table.read(catfile, hdu=2)
        ccat = SkyCoord(ra=tilecat["ALPHA_J2000"],
                        dec=tilecat["DELTA_J2000"])
        for phot in config["sex_phot"]:
            zptable = zptables[phot]
            errkey = errkeys[phot]
            idx = np.where((zptable["OBJECT"]==tile) &
                           (zptable["FILTER"]==band))[0]
            if len(idx) == 0:
                continue
            zps = zptable[idx]
            ################################################################
            # Read and calibrates single catalogs
            cats = []
            for cat in zps:
                catname = os.path.join(config["main_dir"], "single",
                                       cat["NIGHT"], cat["FILENAME"])
                data = Table.read(catname, hdu=2)
                idx = np.where(data["FLAGS"]==0)[0]
                data = data[idx]
                # Apply calibration
                z = np.power(10, -0.4 * cat["ZP"])
                zerr = z * 0.4 * np.log(10.) * cat["ZPERR"]
                f0, f0err =  data[phot], data[errkey]
                f = f0 * z
                f0err = np.sqrt((f0err * z)**2 + (f0 * zerr)**2)
                data[phot] = f
                data[errkey] = f0err
                cats.append(data)
            ################################################################
            refcat = make_reference_catalog(cats, phot, errkey)
            # Selecting stars in the stellar locus
            for snmin in [20, 10, 5]:
                sn = refcat[phot] / refcat[errkey]
                nstars = len(np.where(sn >= snmin)[0])
                if nstars > 100:
                    refcat = refcat[sn >= snmin]
                    break
            radius = refcat["FLUX_RADIUS"]
            rmean, rmedian, rstd = sigma_clipped_stats(radius, sigma=3.)
            refcat = refcat[refcat["FLUX_RADIUS"] <= rmedian + 3 * rstd]
            refcat = refcat[refcat["FLUX_RADIUS"] >= rmedian - 3 * rstd]
            ################################################################
            # Aligning reference and tile catalogs for diff. photometry
            cref = SkyCoord(ra=refcat["ALPHA_J2000"],
                            dec=refcat["DELTA_J2000"])
            idx, d2d, d3d = cref.match_to_catalog_sky(ccat)
            goodidx = np.where(d2d.to("arcsec").value < 0.5)[0]
            cdata = tilecat[idx][goodidx]
            rdata = refcat[goodidx]
            ################################################################
            # Determination of the zero point using total aperture fluxes
            ratio = rdata[phot] / cdata[phot]
            ratioerr = np.sqrt(
                   np.power(rdata[errkey] / cdata[phot], 2) +
                   np.power(ratio * cdata[errkey] / cdata[phot], 2))
            weights = 1 / ratioerr**2
            if weights.sum() == 0:
                print("Weights equal to zero, skipping")
                continue
            weights /= weights.sum()
            f0 = np.average(ratio, weights = weights)
            f0err = np.sqrt(np.average(np.power(ratio - f0, 2),
                                       weights=weights))
            m0 = -2.5 * np.log10(f0)
            m0err = np.abs(2.5 / np.log(10) * f0err / f0)
            t = Table()
            t["TILE"] = [tile]
            t["GAINTABLE"] = [config["name"]]
            t["FILTER"] = [band]
            t["FLUXTYPE"] = [phot]
            t["ZP"] = [round(m0, 5)]
            t["ZPERR"] = [round(m0err, 5)]
            results.append(t)
    if len(results) == 0:
        return
    results = vstack(results)
    results.write(output, overwrite=True)
    return

def main(survey=None):
    # Check if configuration file is given, or use default ones.
    config_files = [_ for _ in sys.argv if _.endswith(".yaml")]
    if len(config_files) == 0:
        print("Using default config files")
        config_files = ["config_mode0.yaml", "config_mode5.yaml"]
    # Loop over all configuration files
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.load(f)
        ########################################################################
        # Setting up directories
        cats_dir = os.path.join(config["main_dir"], "tiles")
        tmp_dir = os.path.join(config["main_dir"], "tmp")
        out_dir = os.path.join(config["main_dir"], "zeropoints/tiles")
        for dir_ in [cats_dir, tmp_dir, out_dir]:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
        config["cats_dir"] = cats_dir
        config["tmp_dir"] = tmp_dir
        config["out_dir"] = out_dir
        ########################################################################
        # Filter tiles to those with calibration
        tiles_with_zps = []
        zptables = {}
        for phot in config["sex_phot"]:
            tablename = os.path.join(config["main_dir"], "zeropoints/single",
                                     "{}-{}.fits".format(config["name"], phot))
            zptable = Table.read(tablename)
            objects = zptable["OBJECT"]
            objects = [obj.replace("_", "-").replace(" ", "") for obj in
                       objects]
            zptable["OBJECT"] = objects
            zptables[phot] = zptable
            tiles_with_zps.append(np.unique(zptable["OBJECT"]))
        tiles_with_zps = set([item for sublist in tiles_with_zps for item in
                              sublist])
        tiles = os.listdir(config["tiles_dir"])  # All reduced tiles
        tiles = sorted([t for t in tiles if t in tiles_with_zps])
        ########################################################################
        # Check the bands with data
        bands = [zptables[phot]["FILTER"] for phot in config["sex_phot"]]
        bands = list(set([item for sublist in bands for item in
                          sublist]))
        bands = [band for band in context.bands if band in bands] # Sorting
        ########################################################################
        if survey is not None:
            tiles = [t for t in tiles if t.startswith(survey)]
        ########################################################################
        # for tile in tiles:
        #     calib_tile(config, bands, zptables, tile)
        f = partial(calib_tile, config, bands, zptables)
        pool = Pool(2)
        pool.map(f, tiles)
    if os.path.exists(tmp_dir):
        os.rmdir(tmp_dir)

if __name__ == "__main__":
    main()