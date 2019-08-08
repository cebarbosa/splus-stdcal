# -*- coding: utf-8 -*-
""" 

Created on 15/03/18

Author : Carlos Eduardo Barbosa

Make the calibration of tiles based on single catalogs.

"""

from __future__ import print_function, division

import os
import platform

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

import context

def calib_tiles(survey, flux_keys, redo=False, catalog_type="single"):
    """ Performs the calibration of catalogs from single exposure. """
    if platform.node() == "kadu-Inspiron-5557":
        tiles_dir = os.path.join(context.data_dir, "tiles_example")
        single_dir = os.path.join(context.data_dir, "single")
    else:
        tiles_dir = "/mnt/jype/MainSurvey/tiles/T01"
        single_dir = "/mnt/public/kadu/stdcal/data/single"
    ############################################################################
    # Setting up reference catalogs
    tiles, bands, paths, nights = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(single_dir):
        for filename in [f for f in filenames if f.endswith(".fits")]:
            paths.append(os.path.join(dirpath, filename))
            bands.append(filename.split("_")[-1].replace(".fits", ""))
            tiles.append(filename.split("_")[-2])
            nights.append(dirpath.split("/")[-1])
    reftable = Table(data=[tiles, bands, paths, nights],
                     names=["tile", "filter", "path", "night"])
    ############################################################################
    # Use only data for selected survey
    idx = [i for i,tile in enumerate(reftable["tile"]) if tile.startswith(
           survey)]
    reftable = reftable[idx]
    ############################################################################
    output_basedir = os.path.join(context.data_dir, "tiles")
    outdir_zps = os.path.join(context.data_dir, "tiles_zps")
    if not os.path.exists(output_basedir):
        os.mkdir(output_basedir)
    if not os.path.exists(outdir_zps):
        os.mkdir(outdir_zps)
    ###########################################################################
    survey_tiles = [_ for _ in os.listdir(tiles_dir) if _.startswith(survey)]
    # Getting tiles catalogs and apply calibration
    for ii, tile in enumerate(survey_tiles):
        print("Tile {} ({}/{})".format(tile, ii+1, len(survey_tiles)))
        tile_dir = os.path.join(tiles_dir, tile)
        for band in context.bands:
            if band not in os.listdir(tile_dir):
                continue
            outdir = os.path.join(output_basedir, tile)
            output = os.path.join(outdir, "{0}_{1}_{2}_catalog.fits".format(
                                   tile, band, catalog_type))
            output2 = os.path.join(outdir_zps, "{}_{}_{}.fits".format(tile,
                                   band, catalog_type))
            if os.path.exists(output) and not redo:
                continue
            idx = np.intersect1d(np.where(tile == reftable["tile"])[0],
                  np.where(band == reftable["filter"])[0])
            refs = reftable[idx]
            refdata = make_reference_catalog(refs, flux_keys)
            if refdata is None:
                continue
            catname = os.path.join(tile_dir, band,
                  "{0}_{1}files/{0}_{1}_{2}.catalog".format(tile, band,
                                                           catalog_type))
            if not os.path.exists(catname):
                continue
            ####################################################################
            # Aligning reference and tile catalogs for differential photometry
            catdata = Table.read(catname, hdu="LDAC_OBJECTS")
            cref = SkyCoord(ra=refdata["ALPHA_J2000"] * u.degree,
                            dec=refdata["DELTA_J2000"] * u.degree)
            try:
                ccat = SkyCoord(ra=catdata["ALPHA_J2000"] * u.degree,
                                dec=catdata["DELTA_J2000"] * u.degree)
            except:
                ccat = SkyCoord(ra=catdata["ALPHA_J2000"],
                                dec=catdata["DELTA_J2000"])
            idx, d2d, d3d = cref.match_to_catalog_sky(ccat)
            goodidx = np.where(d2d.to("arcsec").value < 0.5)[0]
            subcatdata = catdata[idx][goodidx]
            subrefdata = refdata[goodidx]
            zps = None
            ####################################################################
            for fluxkey in flux_keys:
                ferrkey = "FLUXERR_APERCOR" if fluxkey == "FLUXAPERCOR" else \
                    fluxkey.replace("FLUX", "FLUXERR")
                magkey = 'MAGAPERCOR' if fluxkey == "FLUXAPERCOR" else \
                    fluxkey.replace("FLUX", "MAG")
                merrkey = "MAGERR_APERCOR" if magkey == "MAGAPERCOR" else \
                    magkey.replace("MAG", "MAGERR")
                # Check if calibration is finite
                if np.all(np.isnan(refdata[fluxkey])):
                    print("Skiping {}: all nans. ".format(fluxkey))
                    catdata[fluxkey] = np.nan
                    catdata[ferrkey] = np.nan
                    catdata[magkey] = np.nan
                    catdata[merrkey] = np.nan
                    continue
                # Limiting the analysis to safe magnitudes
                magmin = np.percentile(catdata[magkey][np.isfinite(
                                                       catdata[magkey])], 2)
                magmax = np.percentile(catdata[magkey][np.isfinite(
                                                       catdata[magkey])], 8)
                idx = np.intersect1d(np.where(subrefdata[magkey] > magmin)[0],
                                     np.where(subrefdata[magkey] < magmax)[0])
                cdata = subcatdata[idx]
                rdata = subrefdata[idx]
                # Determination of the zero point using total aperture fluxes
                if not fluxkey in cdata.colnames:
                    continue
                ratio = rdata[fluxkey] / cdata[fluxkey]
                ratioerr = np.sqrt(
                       np.power(rdata[ferrkey] / cdata[fluxkey], 2) +
                       np.power(ratio * cdata[ferrkey] / cdata[fluxkey], 2))
                weights = 1 / ratioerr**2
                if weights.sum() == 0:
                    print("Weights equal to zero, skipping")
                    continue
                weights /= weights.sum()
                f0 = np.average(ratio, weights = weights)
                f0err = np.sqrt(np.average(np.power(ratio - f0, 2),
                                           weights=weights))
                ################################################################
                # Determination of the zero point using total aperture fluxes
                diff = rdata[magkey] - cdata[magkey]
                differr = np.sqrt(np.power(rdata[merrkey], 2))
                weights = np.power(differr, -2)
                weights /= weights.sum()
                m0 = np.average(diff, weights=weights)
                m0err =  np.sqrt(np.average(np.power(diff - m0, 2),
                                        weights=weights))
                #################################################################
                # Applying zero points to data
                f = catdata[fluxkey]
                ferr = catdata[ferrkey]
                catdata[fluxkey] = f * f0
                catdata[ferrkey] = np.sqrt(np.power(f * f0err, 2) +
                                           np.power(f0 * ferr, 2))
                m = catdata[magkey]
                merr = catdata[merrkey] if merrkey in catdata.colnames else 0.
                catdata[magkey] = m + m0
                catdata[merrkey] = np.sqrt(m0err**2 + merr**2)
                zp = Table([[fluxkey], [m0 + 25.], [m0err]],
                           names=["phot", "zp", "zperr"])
                if zps is None:
                    zps = zp
                else:
                    zps = vstack([zps, zp])
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            if zps is not None:
                catdata.write(output, overwrite=True, format="fits")
                zps.write(output2, overwrite=True, format="fits")


def mag(f):
    return np.log10(f)

def make_reference_catalog(cats, flux_keys):
    """ Select a catalog among a list to be used for differencial photometry."""
    if not len(cats):
        return None
    elif len(cats) == 1:
        return Table.read(cats[0]["path"], hdu="LDAC_OBJECTS")
    data = [Table.read(cat["path"], hdu="LDAC_OBJECTS") for cat in cats]
    coords = [SkyCoord(ra=cat["ALPHA_J2000"] * u.degree,
                       dec=cat["DELTA_J2000"] * u.degree) for cat in data]
    ############################################################################
    # Aligning catalogs
    newcats = []
    for i, (c, cat) in enumerate(zip(coords, data)):
        if i == 0:
            newcats.append(cat)
            continue
        c0 = SkyCoord(ra=newcats[0]["ALPHA_J2000"] * u.degree,
                      dec=newcats[0]["DELTA_J2000"] * u.degree)
        ci = SkyCoord(ra=cat["ALPHA_J2000"] * u.degree,
                      dec=cat["DELTA_J2000"] * u.degree)
        idx, d2d, d3d = c0.match_to_catalog_sky(ci)
        goodidx = np.where(d2d.to("arcsec").value < 0.5)[0]
        newcat = cat[idx][goodidx]
        newcats.append(newcat)
        for j in range(len(newcats) -1):
            newcats[j] = newcats[j][goodidx]
    ############################################################################
    # Calibrating fluxes
    newcat = Table.copy(newcats[0])
    for fluxkey in flux_keys:
        ferrkey = "FLUXERR_APERCOR" if fluxkey == "FLUXAPERCOR" else \
            fluxkey.replace("FLUX", "FLUXERR")
        magkey = 'MAGAPERCOR' if fluxkey == "FLUXAPERCOR" else \
            fluxkey.replace("FLUX", "MAG")
        merrkey = "MAGERR_APERCOR" if magkey == "MAGAPERCOR" else \
            magkey.replace("MAG", "MAGERR")
        # Calibrating in flux
        flux = np.dstack([dat[fluxkey].quantity.value for dat in newcats])
        errors = np.dstack([dat[ferrkey].quantity.value for dat in
                          newcats])
        weights = np.power(errors, -2)
        meanf = np.average(flux, weights=weights, axis=-1)
        meanferr = np.sqrt(np.sum(np.power(weights * errors, 2), axis=-1)) / \
                           np.sum(weights, axis=-1)
        newcat[fluxkey] = meanf
        newcat[ferrkey] = meanferr
        # Calibrating magnitudes
        mags = np.dstack([dat[magkey] for dat in newcats])
        magerr = np.dstack([dat[merrkey] for dat in newcats])
        weights = np.power(magerr, -2)
        meanm = np.average(mags, weights=weights, axis=-1)
        merr = np.sqrt(np.sum(np.power(weights * magerr, 2), axis=-1)) / \
                       np.sum(weights, axis=-1)
        newcat[magkey] = meanm
        newcat[merrkey] = merr
    return newcat


if __name__ == "__main__":
    flux_keys = ["FLUX_AUTO", "FLUX_ISO", 'FLUXAPERCOR']
    calib_tiles("STRIPE82", flux_keys, redo=True)
