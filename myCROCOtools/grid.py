#!/usr/bin/env python

import numpy as np
import xarray as xr


#--------------
# Vertical mesh
#--------------
def grid_z(ds, tvar=False):
  '''
  Compute vertical mesh of simulation data and add it to the xarray ds.

  Parameters:
	- ds: the data used to computed vertical mesh
	- tvar: if True, compute time evovling vertical mesh.
		(default=False)
  '''

  [nt, nr, ny, nx] = [ds.dims['time'], ds.dims['s_rho'], ds.dims['eta_rho']-2, ds.dims['xi_rho']-2]

  zzt = ds.hc * ds.sc_r + ds.Cs_r*ds.h
  zzw = ds.hc * ds.sc_w + ds.Cs_w*ds.h
  if tvar:
    print('-- Compute time variable vertical mesh --')
    z_rho = (1.0 *(ds.zeta*(1+zzt) + ds.h*zzt)/(ds.h+ds.hc)).transpose('time', 's_rho', 'eta_rho', 'xi_rho')
    z_w   = (1.0 *(ds.zeta*(1+zzw) + ds.h*zzw)/(ds.h+ds.hc)).transpose('time', 's_w', 'eta_rho', 'xi_rho')
    #
    dz_rho    = xr.zeros_like(z_rho)
    dz_rho[:] = z_w[:, 1:, ...].data-z_w[:, :-1, ...].data
    dz_w      = xr.zeros_like(z_w)
    dz_w[:, 1:-1, ...] = z_rho[:, 1:, ...]-z_rho[:, :-1, ...].data
    dz_w[:, 0, ...]    = (z_rho[:, 1, ...]-z_rho[:, 0, ...].data)/2
    dz_w[:, -1, ...]   = (z_rho[:, -1, ...]-z_rho[:, -2, ...].data)/2 
    
  else:
    print('-- Compute vertical mesh at rest --')
    z_rho = (1.0*ds.h*zzt/(ds.h+ds.hc)).transpose('s_rho', 'eta_rho', 'xi_rho')
    z_w   = (1.0*ds.h*zzw/(ds.h+ds.hc)).transpose('s_w', 'eta_rho', 'xi_rho')
    #
    dz_rho    = xr.zeros_like(z_rho)
    dz_rho[:] = z_w[1:, ...].data-z_w[:-1, ...].data
    dz_w      = xr.zeros_like(z_w)
    dz_w[1:-1, ...] = z_rho[1:, ...]-z_rho[:-1, ...].data
    dz_w[0, ...]    = (z_rho[1, ...]-z_rho[0, ...].data)/2
    dz_w[-1, ...]   = (z_rho[-1, ...]-z_rho[-2, ...].data)/2

  ds['z_rho'] =z_rho
  ds['z_w']   =z_w
  ds['dz_rho']=dz_rho
  ds['dz_w']  =dz_w
 
  return ds


#----------------
# horizontal mesh
#----------------
def grid_hz(ds):
  '''
  Compute horizontal mesh of curvilinear grid for rho, u, v, psi points
  '''

  if 'pm' in ds and 'pn' in ds:
    print('-- Compute horizontal mesh --')
    ds['dx_rho'] = 1/ds['pm']
    ds['dy_rho'] = 1/ds['pn']
    ds['dx_u']   = (['eta_rho', 'xi_u'], \
                    0.5 *(1/ds['pm'][:, 1:].data+1/ds['pm'][:, :-1].data) )
    ds['dy_u']   = (['eta_rho', 'xi_u'], \
                    0.5 *(1/ds['pn'][:, 1:].data+1/ds['pn'][:, :-1].data) )
    ds['dx_v']   = (['eta_v', 'xi_rho'], \
                    0.5 *(1/ds['pm'][1:, :].data+1/ds['pm'][:-1, :].data) )
    ds['dy_v']   = (['eta_v', 'xi_rho'], \
                    0.5 *(1/ds['pn'][1:, :].data+1/ds['pn'][:-1, :].data) )
    ds['dx_psi'] = (['eta_v', 'xi_u']  , \
                    0.25*( 1/ds['pm'][1: , 1: ].data+1/ds['pm'][1: , :-1].data\
                          +1/ds['pm'][:-1, :-1].data+1/ds['pm'][:-1, 1: ]).data )
    ds['dy_psi'] = (['eta_v', 'xi_u']  , \
                    0.25*( 1/ds['pn'][1: , 1: ].data+1/ds['pn'][1: , :-1].data\
                          +1/ds['pn'][:-1, :-1].data+1/ds['pn'][:-1, 1: ].data) )
  else:
    print('pm and pn not in ds ; can not recomputed horizontal mesh')

  return ds
        
