#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import find_peaks   # to find mixed layer depth
import xrft                           # xarray fourier transform

from .grid import *


def mld(da)
  '''
  Compute the depth of the mixed layer based on horizontally averaged density profil.
  '''

  #-- constant --
  g=9.81
  rho0=1024

  #-- dimension and grid --
  [nt, nr, ny, nx] = [da.dims['time'], da.dims['s_rho'], da.dims['eta_rho']-2, da.dims['xi_rho']-2]
  [deptht, depthw] = grid_z(da)


  for iit in range(nt):
    print( iit, "/", nt, end="\r")
    #
    rho_m = nbq.rho[iit, :, 1:-1, 1:-1].mean('xi_rho').mean('eta_rho')
    N2_m  = -grav/rhoRef * (rho_m[1:None].data-rho_m[:-1].data)/dz 
    # find peaks (à la Guillaume!) 
    peaks, _ = find_peaks(N2_m, height=strat)
    kmld[0, iit] = int(peaks[-1]+1)
    mld[0, iit] = zzw1d[peaks[-1]+1]
