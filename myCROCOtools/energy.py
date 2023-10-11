#!/usr/bin/env python

import numpy as np
import xarray as xr
from scipy.signal import find_peaks   # to find mixed layer depth
import xrft                           # xarray fourier transform


#---------------
# Kinetic energy
#---------------
def ke(ds, units='J'):
  '''
  Compute kinetic energy as (defined at rho-points):
  
                        _____I  _____J
                 1    /     2       2   \
           KE = --- * |   U    +  V     |
                 2    \                 /

         __I     __J
  where   .  and  .  represent xi- and eta- spatial interpolation 
  (which do not include grid cell area weighting following diag.F)

  Parameters:
        - ds: the data used to computed vertical mesh
        - units: either in Joules ('J', default) or in m^2/s^2 (i.e. the specific energy = per unit mass; 'spec')
  '''

  #-- constant --
  rho0=1024

  if 's_w' in ds.w.dims:
    nbq=True
  else:
    nbq=False

  if not 'time' in ds.dz_rho or 'time' in ds.dz_w:
    print("!!! Vertical mesh dz_rho/dz_w are constant in time (computed at rest). !!!")
    print("!!! Computation of KE would be more precise if dz_rho/dz_w are time varying, !!!")
    print("!!! if they account for free surface elevation. !!!")

  #
  ke=xr.zeros_like(ds.temp)
  #- horizontal part -
  ke[:, :, 1:-1, 1:-1] = 0.25*( ds.u[:, :, 1:-1, :-1 ].data**2 + ds.u[:, :, 1:-1, 1:  ].data**2 \
                               +ds.v[:, :, :-1 , 1:-1].data**2 + ds.v[:, :, 1:  , 1:-1].data**2 )
  #- vertical part -
  if nbq:
    ke[:, :, 1:-1, 1:-1] = ke[:, :, 1:-1, 1:-1] + \
                           0.25*(ds.w[:, :-1, 1:-1, 1:-1].data**2 + ds.w[:, 1:, 1:-1, 1:-1].data**2)
  else:
    ke[:, :, 1:-1, 1:-1] = ke[:, :, 1:-1, 1:-1] + 0.5*ds.w[:, :, 1:-1, 1:-1].data

  #- units -
  if units=='J' :
    print('KE is in Joules')
    ke=rho0*ke*ds.dx_rho*ds.dy_rho*ds.dz_rho
  elif units=='spec':
    print(r'KE is in m$^2$s$^{-1}$')
  else:
    print('Does not knoe units: %s' % units)
    print("Please provide either 'J' or 'spec'.")
    sys.exit()

  return ke
