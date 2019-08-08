"""
Created on Dec 26, 2017

Author: Carlos Eduardo Barbosa

Find catalogs of standard stars and retrieve information for the determination
of instrumental magnitudes.

"""
from __future__ import print_function, division

import os

import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from tqdm import tqdm

import context

def search_standards(redo=True):
    """ Search for catalogs and get catalog information for standard stars. """
    reduced_path = "/mnt/jype/MainSurvey/reduced"
    output_root = "/mnt/public/kadu/stdcal/data/extmoni"
    nights = [_ for _ in os.listdir(reduced_path) if
              os.path.isdir(os.path.join(reduced_path, _))]
    for night in nights:
        outdir = os.path.join(output_root, night)
        if os.path.exists(outdir) and not redo:
            continue
        path = os.path.join(reduced_path, night)
        images = [_ for _ in os.listdir(path) if _.endswith("_proc.fz")]
        images = sorted([_ for _ in images if _.startswith("EXTMONI")])
        cats = [os.path.join(path, _.replace(".fz", "files"),
                             _.replace(".fz", ".apercorcat")) for _ in images]
        for img, cat in zip(images, cats):
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            if not os.path.exists(cat):
                continue
            output = os.path.join(outdir, "{}.fits".format(
                img.split("_proc")[0]))
            if os.path.exists(output) and not redo:
                continue
            h = fits.getheader(os.path.join(path, img), 1)
            obj = h["object"]
            q = Vizier.query_object(obj, catalog="I/256/nltt")[0]
            starcoords = SkyCoord(q["RAJ2000"], q["DEJ2000"],
                                  unit=(u.hourangle, u.degree))
            catdata = Table.read(cat, hdu=2)
            catcoords = SkyCoord(catdata["ALPHA_J2000"],
                                 catdata["DELTA_J2000"],
                                 unit=(u.degree, u.degree))
            idx, d2d, d3d = starcoords.match_to_catalog_sky(catcoords)
            stardata = catdata[idx]
            t = Table()
            t["IMAGE"] = [img.split("_proc")[0]]
            t["DATE"] = [night]
            t["STAR"] = [obj]
            t["FILTER"] = h["filter"]
            t["AIRMASS"] = h["airmass"]
            t["EXPTIME"] = h["exptime"]
            t["D2D"] = [d2d.to("arcsec")]
            cat = hstack([stardata, t])
            cat.write(output, format="fits", overwrite=True)

def join_tables(redo=True):
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
    get_std_data()
    join_tables()