#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import find_peaks   # to find mixed layer depth
import xrft                           # xarray fourier transform

from .grid import *


#------------------
# mixed layer depth
#------------------
def mld(ds, strat=1.9620001275490499e-6):
  '''
  Compute the depth of the mixed layer based on horizontally averaged density profile.

  Parameters:
        - ds: the data used to computed vertical mesh
        - strat: Initial stratification to avoid peaks of N2 smaller than initial stratification
		(typically near the surface).
  Output:
	- mld: depth of the mixed layer (at w-point, where N2 is computed)
	- kmld: associated vertical grid index 
  '''

  #-- constant --
  g=9.81
  rho0=1024

  #-- dimension and grid --
  [nt, nr, ny, nx] = [ds.dims['time'], ds.dims['s_rho'], ds.dims['eta_rho']-2, ds.dims['xi_rho']-2]
  if 'time' in ds.dz_w.dims:
    print('-- Vertical mesh is time variable ; only consider first time step')
    zw_1d =  ds.z_w[0, :, 1:-1, 1:-1].mean(dim=['xi_rho','eta_rho'])
    dz_1d = ds.dz_w[0, :, 1:-1, 1:-1].mean(dim=['xi_rho','eta_rho'])
  else:
    zw_1d =  ds.z_w[:, 1:-1, 1:-1].mean(dim=['xi_rho','eta_rho'])
    dz_1d = ds.dz_w[:, 1:-1, 1:-1].mean(dim=['xi_rho','eta_rho'])

  mld  = np.zeros(nt)
  kmld = np.zeros(nt)
  for iit in range(nt):
    print('-- Compute mld for time: ', iit, "/", nt, end="\r")
    #
    rho_m = ds.rho[iit, :, 1:-1, 1:-1].mean(dim=['xi_rho','eta_rho'])
    N2_m  = xr.zeros_like(zw_1d)
    N2_m[1:-1] = -g/rho0 * (rho_m[1:None].data-rho_m[:-1].data)/dz_1d[1:-1] 
    # find peaks (à la Guillaume!) 
    peaks, _ = find_peaks(N2_m, height=strat)
    kmld[iit] = peaks[-1] 
    mld[iit]  = zw_1d[peaks[-1]]

  return mld, kmld


#------------------
# buoyancy fluxes
#------------------
def wb(ds, qnet=-500, tserie='last', ttt=-1):
  '''
  Compute vertical buoyancy fluxes

  Parameters:
        - ds: the data 
	- qnet: surface net heat flux [W/m^2]
	- tserie: 'last' (default) -> last time record of ds
		  'full'           -> full time series
	-         'partial'        -> only ttt time records
  '''

  #-- constant --
  rho0   = 1024.              # [kg/m3] -- from croco.in
  g      = 9.81               # [m/s2]  -- from croco
  alphaT = 2.e-4              # [K^{-1}]-- from croco.in
  Cp     = 3985.              # [J K^{-1} kg^{-1}] -- from scalar.h
 
  #--
  wb0 = (g*alphaT*qnet)/(rho0*Cp)
  bbb = (ds.rho - (rho0-1000)) *g / rho0
  wb  = xr.zeros_like(ds.w)
  if tserie == 'full': 
    wb               = xr.zeros_like(ds.w)
    wb[:, -1, ...]   = wb0
    wb[:, 1:-1, ...] = ds.w[:, 1:-1, ...] * \
                       0.5*(bbb[:, 1:, ...].data+bbb[:, :-1, ...].data)
  elif tserie == 'partial':
    nt = len(ttt)
    wb               = xr.zeros_like(ds.w[ttt, ...])
    wb[:, -1, ...]   = wb0
    wb[:, 1:-1, ...] = ds.w[ttt, 1:-1, ...] * \
                       0.5*(bbb[ttt, 1:, ...].data+bbb[ttt, :-1, ...].data)
  elif tserie == 'last':
    wb               = xr.zeros_like(ds.w[-1, ...])
    wb[-1, ...]   = wb0
    wb[1:-1, ...] = ds.w[-1, 1:-1, ...] * \
                    0.5*(bbb[-1, 1:, ...].data+bbb[-1, :-1, ...].data)
  else:
    print("-- ERROR: please specify tserie properly (full, last, partial)")
    sys.exit()

  return wb

