#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import find_peaks   # to find mixed layer depth
import xrft                           # xarray fourier transform


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
def wb(ds, tracer='b', qnet=500, tserie='last'):
  '''
  Compute vertical fluxes for tracer 'tracer'

  Parameters:
        - ds: the data 
        - tracer: 't' temperature.
                  's' salinity
                  'b' (default), buoyancy (requires an EOS ; only linear, temperature driven coded for now (10/11/2023))
	- qnet: surface net heat flux [W/m^2]. Atm convention, +=up.
	- tserie: 'full'           -> full time series
	-         'last' (default) -> only last time record

  Output:
	- wb, computed at cell interface (omega points) for both
          nbq AND hydrostatic simulations. 
          For the latter, wb is first computed at rho-point and then interpolated.
          Surface boundary condition is specified through qnet.

  Comments:
        - reference density, temperature (and salinity) are recomputed 
          as the basin averaged quantities.
  '''

  #-- constant --
  alphaT = 2.e-4              # [K^{-1}]-- from croco.in
  Cp     = 3985.              # [J K^{-1} kg^{-1}] -- from scalar.h
  g      = 9.81               # [m/s2]  -- from croco

  #--
  if tracer == 'b':
    print('-- compute BUOYANCY vertical flux --')
    rho0 = 1000 + (ds.rho[0, :, 1:-1, 1:-1]*ds.dx_rho[1:-1, 1:-1]*ds.dy_rho[1:-1, 1:-1]*ds.dz_rho[:, 1:-1, 1:-1]).sum() \
                 /(ds.dx_rho[1:-1, 1:-1]*ds.dy_rho[1:-1, 1:-1]*ds.dz_rho[:, 1:-1, 1:-1]).sum()
    wb0 = (g*alphaT*qnet)/(rho0*Cp)
    bbb = -(ds.rho+1000 - rho0)*g / rho0
  elif tracer == 't':
    print('-- compute TEMPERATURE vertical flux --')
    T0 = (ds.temp[0, :, 1:-1, 1:-1]*ds.dx_rho[1:-1, 1:-1]*ds.dy_rho[1:-1, 1:-1]*ds.dz_rho[:, 1:-1, 1:-1]).sum() \
        /(ds.dx_rho[1:-1, 1:-1]*ds.dy_rho[1:-1, 1:-1]*ds.dz_rho[:, 1:-1, 1:-1]).sum()
    wb0 = qnet/(rho0*Cp)
    bbb = ds.temp-T0
  elif tracer == 's':
    print('-- compute SALINITY vertical flux --')
    print('-->> TO BE DONE <<--')
    sys.exit()
 
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


#-----------
# KE spectra
#-----------
def kespect(ds, tserie='last', ttt=[-1]):
  '''
  Compute 2D isotropic KE power spectra, 
  assuming isotropic and hoogeneous horizontal resolution.

  Parameters:
	- ds: the data
	- tserie: 'full'             -> full time series
	-         'partial' (defalt) -> only ttt time records
  '''

  #-- parameters --
  dxy = ds.dx_rho[1:-1, 1:-1].mean(dim=['eta_rho', 'xi_rho']).data
  if 's_w' in ds.w.dims:
    nbq=True
  else:
    nbq=False

  #--
  if tserie=='full':
    nt=ds.dims['time']
    ttt=np.arange(nt)
  else:
    nt=len(ttt)

  ds_spectra={}
  for it in range(nt):
    da = xr.Dataset( )
    da["uvel"] = (['z', 'y', 'x'], \
                  0.5*(ds.u[it, :, 1:-1, :-1].data + ds.u[ttt[it], :, 1:-1, 1:].data) )
    da["vvel"] = (['z', 'y', 'x'], \
                  0.5*(ds.v[it, :, :-1, 1:-1].data + ds.v[ttt[it], :, 1:, 1:-1].data) )
    if nbq:
      da["wvel"] = (['z', 'y', 'x'], \
                    0.5*(ds.w[it, :-1, 1:-1, 1:-1].data + ds.w[ttt[it], 1:, 1:-1, 1:-1].data) )
    else:
      da["wvel"] = (['z', 'y', 'x'],  ds.w[ttt[it], :, 1:-1, 1:-1].data )
    #- compute dk -
    u_dft = xrft.dft(da.uvel, \
                 dim=['x', 'y', 'z'], true_phase=True, true_amplitude=False).compute()
    dk = u_dft["freq_x"].spacing
    #- compute spectra -
    ufft = xrft.isotropic_power_spectrum(da.uvel, \
           dim=['x', 'y'], true_phase=True, true_amplitude=False).compute()
    vfft = xrft.isotropic_power_spectrum(da.vvel, \
           dim=['x', 'y'], true_phase=True, true_amplitude=False).compute()
    wfft = xrft.isotropic_power_spectrum(da.wvel, \
           dim=['x', 'y'], true_phase=True, true_amplitude=False).compute()
    ds_spectra[it] = 0.5*( (abs(ufft.data)**2 + abs(vfft.data)**2 + abs(wfft.data)**2) * dk**2 ).sum(axis=0) * dxy

  return ds_spectra, ufft.freq_r
