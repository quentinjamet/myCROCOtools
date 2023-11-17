#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from   xgcm         import Grid
from   scipy.signal import find_peaks   # to find mixed layer depth
import xrft                             # xarray fourier transform


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

  #####################
  #- define xgcm grid -
  if 'CPP-options' in ds.attrs:
      cpp = 'CPP-options'
  else:
      cpp = 'CPPS'
  #
  coords={'x':{'center':'xi_rho',  'left':'xi_u'},
          'y':{'center':'eta_rho', 'left':'eta_v'},
          'z':{'center':'s_rho',   'outer':'s_w'}}
  grid = Grid(ds,
         coords=coords,
         periodic=True)
  #####################

  #-- constant --
  g=9.81
  rho0=1024

  #-- compute --
  ds['N2_m']  = -g/rho0 * grid.diff(ds.rho.mean(dim=['xi_rho','eta_rho']), 'z') / ds.dz_w.mean(dim=['xi_rho','eta_rho'])
  # correct for upper and lower values because xgcm assumed periodic boundary conditions
  ds.N2_m[:,0]   = ds.N2_m[:, -1] = 0.0
  #

  mld  = np.zeros_like(ds.time)
  kmld = np.zeros_like(ds.time)
  for iit in range(ds.dims['time']):
    print('-- Compute mld for time: ', iit, "/", ds.dims['time'], end="\r")
    # find peaks (à la Guillaume!) 
    peaks, _ = find_peaks(ds.N2_m[iit], height=strat)
    kmld[iit] = peaks[-1] 
    if 'time' in ds.z_w.dims:
      mld[iit]  = ds.z_w.mean(dim=['xi_rho','eta_rho'])[iit, peaks[-1]]
    else:
      mld[iit]  = ds.z_w.mean(dim=['xi_rho','eta_rho'])[peaks[-1]]
 
  ds['mld'] = mld
  ds['kmld'] = kmld

  ds = ds.drop_vars(["N2_m"])

  return ds


#------------------
# buoyancy fluxes
#------------------
def wb(ds, tracer='b', sbcs=None, tserie='last'):
  '''
  Compute vertical fluxes for tracer 'tracer'

  Parameters:
        - ds: the data 
        - tracer: 'T' temperature
                  'S' salinity (to be coded)
                  'b' (default), buoyancy (requires an EOS ; only linear, temperature driven coded for now (10/11/2023))
	- sbcs: surface net heat flux [W/m^2]. Atm convention, +=up.
	- tserie: 'full'           -> full time series
	-         'last' (default) -> only last time record

  Output:
	- wb, computed at cell interface (omega points) for both
          nbq AND hydrostatic simulations. 
          For the latter, wb is first computed at rho-point and then interpolated.
          Surface boundary condition is specified through sbcs.

  Comments:
        - reference density, temperature (and salinity) are recomputed 
          as the basin averaged quantities.
	- compressible_NS_paper.ipynb for implementation with xgcm
  '''

  #-- constant --
  alphaT = 2.e-4              # [K^{-1}]-- from croco.in
  Cp     = 3985.              # [J K^{-1} kg^{-1}] -- from scalar.h
  g      = 9.81               # [m/s2]  -- from croco
  rho0   = 1000.0 + (ds.rho[0, ...]*ds.dx_rho*ds.dy_rho*ds.dz_rho).sum() \
                   /(ds.dx_rho*ds.dy_rho*ds.dz_rho).sum()

  #--
  if tracer == 'b':
    print('-- compute BUOYANCY vertical flux --')
    if sbcs != None:
      wb0 = (g*alphaT*sbcs)/(rho0*Cp)
    else:
      wb0 = 0.0
    bbb = -(ds.rho+1000 - rho0)*g / rho0
  elif tracer == 'T':
    print('-- compute TEMPERATURE vertical flux --')
    T0 = (ds.temp[0, ...]*ds.dx_rho*ds.dy_rho*ds.dz_rho).sum() \
        /(ds.dx_rho*ds.dy_rho*ds.dz_rho).sum()
    if sbcs != None:
      wb0 = sbcs/(rho0*Cp)
    else:
      wb0 = 0.0
    bbb = ds.temp-T0
  elif tracer == 'S':
    print('-- compute SALINITY vertical flux --')
    sys.exit("-->> TO BE DONE <<--")
  else:
    sys.exit("==>> expected: b, T, S ; provided: %s" % tracer)
 
  #--
  if tserie=='full':
    nt=ds.dims['time']
    ttt=np.arange(nt)
  #
  if ds.attrs['CPP-options'].find('NBQ')!= -1:
    nbq=True
  else:
    nbq=False

  #--
  if nbq:
    if tserie=='last':
      wb  = xr.zeros_like(ds.w[-1, ...])
      wb[-1, ...]   = wb0
      wb[1:-1, ...] = ds.w[-1, 1:-1, ...] * \
                      0.5*(bbb[-1, 1:, ...].data+bbb[-1, :-1, ...].data)
    else:
      wb  = xr.zeros_like(ds.w)
      wb[:, -1, ...]   = wb0
      wb[:, 1:-1, ...] = ds.w[:, 1:-1, ...] * \
                         0.5*(bbb[:, 1:, ...].data+bbb[:, :-1, ...].data)
  else:
    if tserie=='last':
      tmp = ds.w[-1, ...]*bbb[-1, ...]
      wb  = xr.zeros_like(ds.omega[-1, ...])
      wb[-1, ...] = wb0
      wb[1:-1, ...] = 0.5*(tmp[1:, ...].data+tmp[:-1, ...].data)
    else:
      tmp = ds.w*bbb
      wb  = xr.zeros_like(ds.omega)
      wb[:, -1, ...] = wb0
      wb[:, 1:-1, ...] = 0.5*(tmp[:, 1:, ...].data+tmp[:, :-1, ...].data)

  return wb
