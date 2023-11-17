#!/usr/bin/env python

import numpy as np
import xarray as xr
from   xgcm         import Grid
from   scipy.signal import find_peaks   # to find mixed layer depth
import xrft                           # xarray fourier transform

#---------------
# Kinetic energy
#---------------
def ke(ds, units='J', tserie='last'):
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
	- tserie: either 'full' time serie (default) or 'last' time record
  '''

  #####################
  #- define xgcm grid -
  if 'CPP-options' in ds.attrs:
      cpp = 'CPP-options'
  else:
      cpp = 'CPPS'
  #
  if ds.attrs[cpp].find('EW_PERIODIC')!= -1 and ds.attrs[cpp].find('NS_PERIODIC')!= -1:
    coords={'x':{'center':'xi_rho',  'left':'xi_u'},
            'y':{'center':'eta_rho', 'left':'eta_v'},
            'z':{'center':'s_rho',   'outer':'s_w'}}
    grid = Grid(ds,
            coords=coords,
            periodic=True)
  else:
    coords={'x':{'center':'xi_rho',  'inner':'xi_u'},
            'y':{'center':'eta_rho', 'inner':'eta_v'},
            'z':{'center':'s_rho',   'outer':'s_w'}}
    grid = Grid(ds,
            coords=coords,
            boundary='extend')
  #####################

  #-- constant --

  if 's_w' in ds.w.dims:
    nbq=True
  else:
    nbq=False

  if not 'time' in ds.dz_rho or 'time' in ds.dz_w:
    print("!!! Vertical mesh dz_rho/dz_w are constant in time (computed at rest). !!!")
    print("!!! Computation of KE would be more precise if dz_rho/dz_w are time varying, !!!")
    print("!!! if they account for free surface elevation. !!!")

  #- horizontal part -
  if tserie ==  'full':
    tmpke = grid.interp(ds.u**2, 'x') + grid.interp(ds.v**2, 'y')
    #- vertical part -
    if nbq:
      tmpke = tmpke + grid.interp(ds.w**2, 'z')
    else:
      tmpke = tmpke + ds.w**2
  elif tserie == 'last':
    tmpke = grid.interp(ds.u[-1, ...]**2, 'x') + grid.interp(ds.v[-1, ...]**2, 'y')
    #- vertical part -
    if nbq:
      tmpke = tmpke + grid.interp(ds.w[-1, ...]**2, 'z')
    else:
      tmpke = tmpke + ds.w[-1, ...]**2

  #- units -
  if units=='J' :
    print('KE is in Joules')
    rho0=1024.
    ds['ke']=0.5*rho0*tmpke*ds.dx_rho*ds.dy_rho*ds.dz_rho
  elif units=='spec':
    print(r'KE is in m2/s2')
    ds['ke']=0.5*tmpke
  else:
    print('Does not know units: %s' % units)
    print("Please provide either 'J' or 'spec'.")
    sys.exit()

  return ds


def kespect(ds, tserie='partial', ttt=-1, anoma=False, density=True):
  '''
  Compute 2D (horizontal) isotropic KE power spectra, 
  assuming isotropic and hoogeneous horizontal resolution.
  Velocities are first interpolated onto rho-grid.

  Parameters:
        - ds: the data
        - tserie: 'full'              -> full time series
        -         'partial' (default) -> only ttt time records
        - anoma : remove horizontal mean (default=False)
        - power spectral density (default=True)

  NOTE: xrft is computing spectra based on dimensions provided, 
	which are supposed to be actual (x,y,z) coordinates.
	In Croco, xi_, eta_ are indices ranging from [0, Nx] (and s_ is vertical stretching ranging from [0, 1]). 
	Horizontal frequency spacing needs to be divided by $\Delta x$ 
	and horizontal Fourier coefficients need to be area weigthed, i.e. multiplied by $\Delta x ^2$.
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

  #-- parameters --
  dxy = ds.dx_rho.mean(dim=['eta_rho', 'xi_rho']).data
  if 's_w' in ds.w.dims:
    nbq=True
  else:
    nbq=False

  #--
  if tserie=='full':
    nt=ds.dims['time']
    ttt=np.arange(nt)
  elif tserie=='partial':
    nt=len(ttt)
  else:
    sys.exit("Don't know option tserie=%s" % tserie)

  ds_spectra={}
  for it in range(nt):
    da = xr.Dataset( )
    da['u'] = grid.interp(ds.u[ttt[it], ...],'x')
    da['v'] = grid.interp(ds.v[ttt[it], ...],'y')
    if nbq:
      da['w'] = grid.interp(ds.w[ttt[it], ...],'z')
    else:
      da['w'] = ds.w[ttt[it], ...]
    #- remove horizontal mean -
    if anoma:
      print('-- Remove horizontal mean --')
      da=da-da.mean(dim=['eta_rho', 'xi_rho'])
    # rename dims and turn them in meters (instead of grid index)
    da = da.rename({'s_rho': 'z','eta_rho': 'y', 'xi_rho': 'x'})
    da = da.assign_coords({"y": da.y*dxy, "x": da.x*dxy})
    #- compute dk -
    u_fft = xrft.fft(da.u, \
                 dim=['x', 'y'], true_phase=True, true_amplitude=True).compute()
    dk = u_fft["freq_x"].spacing
    #- compute spectra -
    ufft = xrft.isotropic_power_spectrum(da.u, \
           dim=['x', 'y'], true_phase=True, true_amplitude=True, density=density).compute()
    vfft = xrft.isotropic_power_spectrum(da.v, \
           dim=['x', 'y'], true_phase=True, true_amplitude=True, density=density).compute()
    wfft = xrft.isotropic_power_spectrum(da.w, \
           dim=['x', 'y'], true_phase=True, true_amplitude=True, density=density).compute()
    ds_spectra[it] = 0.5*( (abs(ufft.data)**2 + abs(vfft.data)**2 + abs(wfft.data)**2) * dk**2 ).sum(axis=0)

  return ds_spectra, ufft.freq_r

