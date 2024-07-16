#!/usr/bin/env python

import numpy as np
import xarray as xr
from   xgcm         import Grid
from   scipy.signal import find_peaks   # to find mixed layer depth
import xrft                           # xarray fourier transform

#---------------
# Kinetic energy
#---------------
def ke(ds, units='J', full=False):
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
	- full (default=False): output 3D KE, otherwise basin averaged only.
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
    grid_xy = Grid(ds,
            coords=coords,
            periodic=True)
  else:
    coords={'x':{'center':'xi_rho',  'inner':'xi_u'},
            'y':{'center':'eta_rho', 'inner':'eta_v'},
            'z':{'center':'s_rho',   'outer':'s_w'}}
    grid_xy = Grid(ds,
            coords=coords,
            boundary='extend')
  grid_z = Grid(ds,
            coords=coords,
            periodic=False)
  #####################

  #-- constant --
  if not 'time' in ds.dz_rho or 'time' in ds.dz_w:
    print("!!! Vertical mesh dz_rho/dz_w are constant in time (computed at rest). !!!")
    print("!!! Computation of KE would be more precise if dz_rho/dz_w are time varying, !!!")
    print("!!! if they account for free surface elevation. !!!")

  tmpke = grid_xy.interp(ds.u**2, 'x') \
         +grid_xy.interp(ds.v**2, 'y') \
         +grid_z.interp(ds.omega**2, 'z')

  #- units -
  if units=='J' :
    print('KE is in Joules')
    rho0=1024.
    if full:
      ds['ke']=0.5*rho0*tmpke*ds.dx_rho*ds.dy_rho*ds.dz_rho
    else:
      ds['ke']=0.5*(rho0*tmpke*ds.dx_rho*ds.dy_rho*ds.dz_rho).sum(dim=['s_rho', 'eta_rho', 'xi_rho'])
  elif units=='spec':
    print(r'KE is in m2/s2')
    if full:
      ds['ke']=0.5*tmpke
    else:
      ds['ke']=0.5*tmpke.sum(dim=['s_rho', 'eta_rho', 'xi_rho'])
  else:
    print('Does not know units: %s' % units)
    print("Please provide either 'J' or 'spec'.")
    sys.exit()

  return ds


def kespect(ds, tserie='partial', ttt=[-1], anoma=False, density=True, metric='uvw'):
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
	- metric: u,v,w,uv, or uvw (default='uvw')

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
  grid_xy = Grid(ds,
         coords=coords,
         periodic=True)
  grid_z = Grid(ds,
         coords=coords,
         periodic=False)
  #####################

  #-- parameters --
  dxy = ds.dx_rho.mean(dim=['eta_rho', 'xi_rho']).data

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
    da['u'] = grid_xy.interp(ds.u[ttt[it], ...],'x')
    da['v'] = grid_xy.interp(ds.v[ttt[it], ...],'y')
    da['w'] = grid_z.interp(ds.omega[ttt[it], ...],'z')
    #- remove horizontal mean -
    if anoma:
      print('-- Remove horizontal mean --')
      da=da-da.mean(dim=['eta_rho', 'xi_rho'])
    # rename and turn dimensions in meters (instead of grid index)
    da = da.rename({'s_rho': 'z','eta_rho': 'y', 'xi_rho': 'x'})
    dxy = ds.dx_rho.mean(dim=['eta_rho', 'xi_rho']).data
    da = da.assign_coords({"y": ds.y_rho[:, 0].data, "x": ds.x_rho[0, :].data})
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

    #
    if metric == 'u':
      ds_spectra[it] = 0.5*( (abs(ufft.data)**2) * dk**2 ).sum(axis=0)
    elif metric == 'v':
      ds_spectra[it] = 0.5*( (abs(vfft.data)**2) * dk**2 ).sum(axis=0)
    elif metric == 'w':
      ds_spectra[it] = 0.5*( (abs(wfft.data)**2) * dk**2 ).sum(axis=0)
    elif metric == 'uv':
      ds_spectra[it] = 0.5*( (abs(ufft.data)**2 + abs(vfft.data)**2) * dk**2 ).sum(axis=0)
    elif metric == 'uvw':
      ds_spectra[it] = 0.5*( (abs(ufft.data)**2 + abs(vfft.data)**2 + abs(wfft.data)**2) * dk**2 ).sum(axis=0)

  return ds_spectra, ufft.freq_r

