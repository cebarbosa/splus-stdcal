# -*- coding: utf-8 -*-
""" 

Created on 19/08/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Distance

if __name__ == "__main__":
    output = "tables/stdcoords.fits"
    stars = ["cd-32_9927", "eg274", "feige110", "hr4468", "hr7950", "ltt1020",
             "ltt1788", "ltt2415", "ltt3218", "ltt377", "ltt3864", "ltt4364",
             "ltt4816", "ltt6248", "ltt7379", "ltt745", "ltt7987", "ltt9239",
             "ltt9491", "cd-34_241"]
    # Stars with large proper motions that require match using parallax
    hpms = ["ltt3218", "ltt4364"]
    width = u.Quantity(0.1, u.deg)
    height = u.Quantity(0.1, u.deg)
    table = []
    Vizier.TIMEOUT = 60
    for star in stars:
        print(star)
        v = Vizier(columns=["**", "+_r"])
        if star not in hpms:
            gaia2 = v.query_region(star, catalog="I/345/gaia2", radius="30s")[0]
            idx = np.where(gaia2["Plx"] > 0)[0]
            gaia2 = Table(gaia2[idx][0])
            distance = Distance(parallax=gaia2["Plx"])
            gaia2["distance"] = distance
        else:
            if star == "ltt3218":
                gaia2 = v.query_region(star, catalog="I/345/gaia2",
                                       radius="30s")[0]
                idx = np.argmax(gaia2["Plx"])
                gaia2 = Table(gaia2[idx])
                distance = Distance(parallax=gaia2["Plx"])
                gaia2["distance"] = distance
            elif star == "ltt4364":
                coords = SkyCoord(ra="11 45 42.9170659451",
                              dec="-64 50 29.464090822",
                              unit=(u.hourangle, u.degree))
                gaia2 = Table()
                gaia2["RA_ICRS"] = [coords.ra.value] * u.degree
                gaia2["DE_ICRS"] =  [coords.dec.value] * u.degree
                gaia2["Plx"] = [215.7373] * u.mas
                gaia2["pmRA"] = [2661.594] * u.mas / u.yr
                gaia2["pmDE"] = [-344.847] * u.mas / u.yr
                distance = Distance(parallax=gaia2["Plx"])
                gaia2["distance"] = distance
        gaia2["STAR"] = star
        startable = gaia2[["STAR", "RA_ICRS", "DE_ICRS", "Plx", "pmRA",
                                     "pmDE", "distance"]]
        table.append(startable)
    table = vstack(table)
    table.write(output, format="fits", overwrite=True)