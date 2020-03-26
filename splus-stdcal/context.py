# -*- coding: utf-8 -*-
"""

Created on 27/09/2017

@Author: Carlos Eduardo Barbosa

Paths used in this project.

"""
import astropy.units as u
import matplotlib.pyplot as plt


# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
# plt.style.use("seaborn-paper")


ps = 0.55 * u.arcsec / u.pixel


bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I',
         'F861', 'Z']

narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']

broad_bands = ['U', 'G', 'R', 'I', 'Z']

bands_names = {'U' : "$u$", 'F378': "$J378$", 'F395' : "$J395$",
               'F410' : "$J410$", 'F430' : "$J430$", 'G' : "$g$",
               'F515' : "$J515$", 'R' : "$r$", 'F660' : "$J660$",
               'I' : "$i$", 'F861' : "$J861$", 'Z' : "$z$"}