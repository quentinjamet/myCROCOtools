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
  coords_xy={'x':{'center':'xi_rho',  'left':'xi_u'},
             'y':{'center':'eta_rho', 'left':'eta_v'}}
  coords_z={'z':{'center':'s_rho',   'outer':'s_w'}}
  grid_z = Grid(ds,
         coords=coords_z,
         periodic=False)
  #####################

  #-- constant --
  g=9.81
  rho0=1024

  #-- compute --
  ds['N2_m']  = -g/rho0 * grid_z.diff(ds.rho.mean(dim=['xi_rho','eta_rho']), 'z', boundary='extrapolate') / ds.dz_w.mean(dim=['xi_rho','eta_rho'])

  mld  = xr.zeros_like(ds.time)
  kmld = xr.zeros_like(ds.time)
  for iit in range(ds.dims['time']):
    print('-- Compute mld for time: ', iit, "/", ds.dims['time'], end="\r")
    # find peaks (à la Guillaume!) 
    peaks, _ = find_peaks(ds.N2_m[iit], height=strat)
    kmld[iit] = peaks[-1] 
    if 'time' in ds.z_w.dims:
      mld[iit]  = ds.z_w.mean(dim=['xi_rho','eta_rho'])[iit, peaks[-1]]
    else:
      mld[iit]  = ds.z_w.mean(dim=['xi_rho','eta_rho'])[peaks[-1]]
 
  ds["mld"]  = mld
  ds["kmld"] = kmld

  ds = ds.drop_vars(["N2_m"])

  return ds


#------------------
# buoyancy fluxes
#------------------
def wb(ds, tracer='b', sbcs=None, full=False):
  '''
  Compute vertical fluxes for tracer 'tracer', 
  based on \Omega S-coordinate vertical momentum component.

  Parameters:
        - ds: the data 
        - tracer: 'T' temperature
                  'S' salinity (to be coded)
                  'b' (default), buoyancy (requires an EOS ; only linear, temperature driven coded for now (10/11/2023))
	- sbcs: surface net tracer flux. Atm convention, +=up. 
	        For now specify only heat flux in [W/m^2], 
	        buoyancy flux are recomputed accordingly.
        - full: outpt 3D vertical fluxes, otherwise compute hz averaging (default=False)

  Output:
	- resolved and sub-grid-scale vertical fluxes of tracer 'tracer'
	  Computed at cell interface (omega points).
          Surface boundary condition is specified through sbcs
	  and applied to sub-grid-scale fluxes.

  Comments:
        - reference density is recomputed as the basin averaged quantities at initial step,
          following RESET_RHO0 CPP option in croco to minimize Boussinesq errors.
  '''

  #####################
  #- define xgcm grid -
  if 'CPP-options' in ds.attrs:
    cpp = 'CPP-options'
  else:
    cpp = 'CPPS'
  #
  coords_xy={'x':{'center':'xi_rho',  'left':'xi_u'},
        'y':{'center':'eta_rho', 'left':'eta_v'}}
  coords_z={'z':{'center':'s_rho',   'outer':'s_w'}}
  grid_xy = Grid(ds,
       coords=coords_xy,
       periodic=True)
  grid_z = Grid(ds,
       coords=coords_z,
       periodic=False)
  #####################

  #-- constant --
  alphaT = 2.e-4              # [K^{-1}]-- from croco.in
  Cp     = 3985.              # [J K^{-1} kg^{-1}] -- from scalar.h
  g      = 9.81               # [m/s2]  -- from croco
  rho0   = 1000.0 + (ds.rho[0, ...]*ds.dx_rho*ds.dy_rho*ds.dz_rho).sum() \
                   /(ds.dx_rho*ds.dy_rho*ds.dz_rho).sum()

  #--
  if tracer == 'b':
    print('-- compute BUOYANCY vertical flux --')
    print('-- based on linear, temperature driven EOS --')
    if sbcs != None:
      wb0 = (g*alphaT*sbcs)/(rho0*Cp)
    else:
      wb0 = 0.0
    bbb = -(ds.rho+1000 - rho0)*g / rho0
  elif tracer == 'T':
    print('-- compute TEMPERATURE vertical flux --')
    if sbcs != None:
      wb0 = sbcs/(rho0*Cp)
    else:
      wb0 = 0.0
    bbb = ds.temp
  elif tracer == 'S':
    print('-- compute SALINITY vertical flux --')
    sys.exit("-->> TO BE DONE <<--")
  else:
    sys.exit("==>> expected: b, T, S ; provided: %s" % tracer)
  
  #-- resolved vertical fluxes --
  print('-->> Resolved fluxes')
  wb = ds.omega * grid_z.interp(bbb, 'z', boundary='extrapolate')
  # set surface boundary condition to zero 
  wb[:, -1, ...]=0.0

  #-- sgs vertical fluxes (if available) --
  if 'AKt' in ds.keys():
    print('-->> Sub-grid scale fluxes based on AKt in ds ; apply boundary conditions given by sbcs')
    wb_sgs = -ds.AKt * grid_z.diff(bbb, 'z', boundary='extrapolate')/ds.dz_w
    # adjust surface boundary condition (Qnet)
    wb_sgs[:, 0, ...] = 0.
    wb_sgs[:, -1, ...] = wb0
  else:
    print('-->> Cannot compute sub-grid scale fluxes, I need estimates of AKt in ds <<--')
    wb_sgs = xr.zeros_like(wb) 

  #-- horizontal averaging --
  name1=str("w%s" % tracer)
  name2=str("w%s_sgs" % tracer)
  if full:
    ds[name1] = wb
    ds[name2] = wb_sgs
  else:
    ds[name1] = wb.mean(dim=['eta_rho','xi_rho'])
    ds[name2] = wb_sgs.mean(dim=['eta_rho','xi_rho'])

  return ds
