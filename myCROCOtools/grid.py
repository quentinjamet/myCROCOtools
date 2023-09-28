#!/usr/bin/env python

import numpy as np


def grid_z(da, tvar='False')
  '''
  Compute vertical grid of simulation data (in m).

  Parameters:
	- da: the data used to computed vertical mesh
	- tvar: if True, compute time evovling vertical mesh.
		(default=False)
  '''

  [nt, nr, ny, nx] = [da.dims['time'], da.dims['s_rho'], da.dims['eta_rho']-2, da.dims['xi_rho']-2]

  zzt = da.hc * da.sc_r + da.Cs_r*da.h
  zzw = da.hc * da.sc_w + da.Cs_w*da.h
  if tvar:
    deptht = (1.0 *(da.zeta*(1+zzt) + da.h*zzt)/(da.h+da.hc)).transpose('time', 's_rho', 'eta_rho', 'xi_rho')
    depthw = (1.0 *(da.zeta*(1+zzw) + da.h*zzw)/(da.h+da.hc)).transpose('time', 's_rho', 'eta_rho', 'xi_rho')
    
  else:
    deptht = (1.0 *(da.zeta[0,...]*(1+zzt) + da.h*zzt)/(da.h+da.hc)).transpose('s_rho', 'eta_rho', 'xi_rho')
    depthw = (1.0 *(da.zeta[0,...]*(1+zzw) + da.h*zzw)/(da.h+da.hc)).transpose('s_rho', 'eta_rho', 'xi_rho')
 
  return deptht, depthw
