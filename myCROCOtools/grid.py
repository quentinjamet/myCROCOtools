#!/usr/bin/env python

import numpy    as np
import xarray   as xr
from   xgcm     import Grid

##----------
## xgcm grid
##----------
#def xgrid(ds):
#  if 'CPP-options' in ds.attrs:
#      cpp = 'CPP-options'
#  else:
#      cpp = 'CPPS'
#
#  if ds.attrs[cpp].find('EW_PERIODIC')!= -1 and ds.attrs[cpp].find('NS_PERIODIC')!= -1:
#    #
#    print('WARNING: In doubly periodic conditions, the 2 (1) additional rho- (u-, v-) grid points in CROCO')
#    print('         are removed since xgcm handles boundary conditions accuratly in this case.')
#    #
#    ds = ds.drop_sel(xi_rho=ds.xi_rho[0]).drop_sel(xi_rho=ds.xi_rho[-1]).drop_sel(eta_rho=ds.eta_rho[0]).drop_sel(eta_rho=ds.eta_rho[-1]).drop_sel(xi_u=ds.xi_u[-1]).drop_sel(eta_v=ds.eta_v[-1])
#    #
#    coords={'x':{'center':'xi_rho',  'left':'xi_u'},
#            'y':{'center':'eta_rho', 'left':'eta_v'},
#            'z':{'center':'s_rho',   'outer':'s_w'}}
#    grid = Grid(ds,
#            coords=coords,
#            periodic=True)
#    ds.attrs['xgcm-Grid'] = Grid(ds, 
#            coords=coords,
#            metrics = metrics,
#            periodic=True)
#
#  else:
#    coords={'x':{'center':'xi_rho',  'inner':'xi_u'},
#            'y':{'center':'eta_rho', 'inner':'eta_v'},
#            'z':{'center':'s_rho',   'outer':'s_w'}}
#    grid = Grid(ds,
#            coords=coords,
#            boundary='extend')
#    ds.attrs['xgcm-Grid'] = Grid(ds, 
#            coords=coords,
#            metrics = metrics,
#            periodic=False,
#            boundary='extend')
#
#  return grid


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

  if 'Cs_rho' in ds:
    zzt = ds.hc * ds.s_rho + ds.Cs_rho*ds.h
  else:
    zzt = ds.hc * ds.s_rho + ds.Cs_r*ds.h
  zzw = ds.hc * ds.s_w + ds.Cs_w*ds.h
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
def grid_hz(ds, vertical=False):
  """
  Create a xgcm grid and set it in the dataset as a attribute.
  For doubly periodic boundary conditions the 2 (1) additional rho- (u-, v-) grid points
  are removed since xgcm handles boundary conditions accuratly in this case.

  Parameters:
      ds : xarray dataset
  returns:
      ds : xarray dataset with the xgcm  grid
      grid : xgcm grid
  """

  print('-- Add horizontal metric --')

  #####################
  #- define xgcm grid -
  if 'CPP-options' in ds.attrs:
      cpp = 'CPP-options'
  else:
      cpp = 'CPPS'
  # Horizontal grid
  if ds.attrs[cpp].find('EW_PERIODIC')!= -1 and ds.attrs[cpp].find('NS_PERIODIC')!= -1:
    coords_xy={'x':{'center':'xi_rho',  'left':'xi_u'}, 
               'y':{'center':'eta_rho', 'left':'eta_v'}}
#               'z':{'center':'s_rho',   'outer':'s_w'}}
    ds = ds.drop_sel(xi_rho=ds.xi_rho[0]).drop_sel(xi_rho=ds.xi_rho[-1]).drop_sel(eta_rho=ds.eta_rho[0]).drop_sel(eta_rho=ds.eta_rho[-1]).drop_sel(xi_u=ds.xi_u[-1]).drop_sel(eta_v=ds.eta_v[-1])
    grid_xy = Grid(ds,
            coords=coords_xy,
            periodic=True)
    print('WARNING: In doubly periodic conditions, the 2 (1) additional rho- (u-, v-) grid points in CROCO')
    print('         are removed since xgcm handles boundary conditions accuratly in this case.')
  else:
    coords_xy={'x':{'center':'xi_rho',  'inner':'xi_u'}, 
               'y':{'center':'eta_rho', 'inner':'eta_v'}}
#            'z':{'center':'s_rho',   'outer':'s_w'}}
    grid_xy = Grid(ds, 
            coords=coords_xy,
            boundary='extend')
  # vertical grid
  coords_z={'z':{'center':'s_rho',   'outer':'s_w'}}
  grid_z = Grid(ds,
            coords=coords_z,
            boundary='extrapolate')

  #####################
  
  
  if 'SPHERICAL' in ds.attrs[cpp]:
      print('- Computes lon/lat at u,v and psi points, and assign to the dataset as coordinates')
      ds['lon_u'] = grid_xy.interp(ds.lon_rho,'x')
      ds['lat_u'] = grid_xy.interp(ds.lat_rho,'x')
      ds['lon_v'] = grid_xy.interp(ds.lon_rho,'y')
      ds['lat_v'] = grid_xy.interp(ds.lat_rho,'y')
      ds['lon_psi'] = grid_xy.interp(ds.lon_v,'x')
      ds['lat_psi'] = grid_xy.interp(ds.lat_u,'y')
      _coords = ['lon_u','lat_u','lon_v','lat_v','lon_psi','lat_psi']
      ds = ds.set_coords(_coords)
      
  if vertical:
      ds['z_u'] = grid_xy.interp(ds.z_rho,'x')
      ds['z_v'] = grid_xy.interp(ds.z_rho,'y')
      _coords = ['z_u','z_v']
      ds = ds.set_coords(_coords)


  # add horizontal distance metrics for (centered at!) rho, u, v and psi point
  if 'pm' in ds and 'pn' in ds:
      ds['dx_rho'] = 1/ds['pm']
      ds['dy_rho'] = 1/ds['pn']
      ds['dx_u'] = grid_xy.interp(1/ds['pm'],'x')
      ds['dy_u'] = grid_xy.interp(1/ds['pn'],'x')
      ds['dx_v'] = grid_xy.interp(1/ds['pm'],'y')
      ds['dy_v'] = grid_xy.interp(1/ds['pn'],'y')
      ds['dx_psi'] = grid_xy.interp(grid_xy.interp(1/ds['pm'], 'y'),  'x') 
      ds['dy_psi'] = grid_xy.interp(grid_xy.interp(1/ds['pn'], 'y'),  'x')
      
  try:
      ds['mask_psi'] = grid_xy.interp(grid_xy.interp(ds.mask_rho, 'y'),  'x') 
  except:
      ds['mask_rho'] = ds['pm']*0.+1.
      ds['mask_psi'] = grid_xy.interp(grid_xy.interp(ds.mask_rho, 'y'),  'x') 


  '''ds.coords['z_rho'][np.isnan(ds.mask_rho)] = 0.
  ds.coords['z_w'][np.isnan(ds.mask_rho)] = 0.
  ds.coords['z_rho'][ds.mask_rho==0] = 0.
  ds.coords['z_w'][ds.mask_rho==0] = 0.'''
  
  if vertical:
      print('- add vertical metrics for u, v, rho and psi points') 
      #ds['dz_rho'] = grid.diff(ds.z_w,'z')
      #ds['dz_w']   = grid.diff(ds.z_rho,'z')
      #ds.ds_w[-1, ...] = (ds.z_rho[-1, ...]-ds.z_rho[-2, ...].data)/2
      #dz_w[0, ...]     = (ds.z_rho[1, ...] -ds.z_rho[0, ...].data)/2
      ds['dz_u']   = grid_xy.interp(ds.dz_rho,'x')
      ds['dz_v']   = grid_xy.interp(ds.dz_rho,'y')
      ds['dz_psi'] = grid_xy.interp(ds.dz_v,'x')
      

  # add areas metrics for rho,u,v and psi points
  #?? should this not be :
  #ds['rArho'] = ds.dx_psi * ds.dy_psi
  #ds['rAu']   = ds.dx_v   * ds.dy_v     
  #ds['rAv']   = ds.dx_u   * ds.dy_u     
  #ds['rApsi'] = ds.dx_rho * ds.dy_rho
  #should this rather be ??
  ds['rArho'] = ds.dx_rho * ds.dy_rho
  ds['rAu']   = ds.dx_u   * ds.dy_u     
  ds['rAv']   = ds.dx_v   * ds.dy_v     
  ds['rApsi'] = ds.dx_psi * ds.dy_psi


#  if vertical:
#      metrics = {
#             ('x',): ['dx_rho', 'dx_u', 'dx_v', 'dx_psi'], # X distances
#             ('y',): ['dy_rho', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
#             ('z',): ['dz_rho', 'dz_u', 'dz_v', 'dz_psi', 'dz_w'], # Z distances
#             ('x', 'y'): ['rArho', 'rAu', 'rAv', 'rApsi'] # Areas
#            }
#  else:
#      metrics = {
#             ('x',): ['dx_rho', 'dx_u', 'dx_v', 'dx_psi'], # X distances
#             ('y',): ['dy_rho', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
#             ('x', 'y'): ['rArho', 'rAu', 'rAv', 'rApsi'] # Areas
#            }
  metrics_xy = {
             ('x',): ['dx_rho', 'dx_u', 'dx_v', 'dx_psi'], # X distances
             ('y',): ['dy_rho', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
             ('x', 'y'): ['rArho', 'rAu', 'rAv', 'rApsi'] # Areas
            }

  if vertical:
      metrics_z = {
             ('z',): ['dz_rho', 'dz_u', 'dz_v', 'dz_psi', 'dz_w'] # Z distances
            }


  if ds.attrs[cpp].find('EW_PERIODIC')!= -1 and ds.attrs[cpp].find('NS_PERIODIC')!= -1:
    ds.attrs['xgcm-Grid_xy'] = Grid(ds, 
            coords=coords_xy,
            metrics = metrics_xy,
            periodic=True)
  else:
    ds.attrs['xgcm-Grid_xy'] = Grid(ds, 
            coords=coords_xy,
            metrics = metrics_xy,
            boundary='extend')
  #
  if vertical:
    ds.attrs['xgcm-Grid_z'] = Grid(ds,
            coords=coords_z,
            metrics = metrics_z,
            boundary='extrapolate')


  return ds


#--------------
# interpolation     
#--------------
def vel_rho(ds):
  '''
  interpolate velocity field at rho-point.
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
  
  ds['u_rho'] = grid.interp(ds.u, 'x')
  ds['v_rho'] = grid.interp(ds.v, 'y')
  ds['w_rho'] = grid.interp(ds.w, 'z')

  return ds


#---------
# Gradient
#---------
#def grad(ds, var='temp', component='x', interp=True):
#    '''
#    Compute horizontal gradient on CROCO grid,
#    interpolated at t-point if interp=True.
#
#    Input:
#        - ds: xarray dataset
#        - var: (default 'temp') variable to compute the gradient
#        - component: direction of the gradient
#        - c_grid: grid location of tmpin
#        - interp: (Default=True) interpolate derivatives at t-points
#    '''
#
#  #####################
#  #- define xgcm grid -
#  if 'CPP-options' in ds.attrs:
#      cpp = 'CPP-options'
#  else:
#      cpp = 'CPPS'
#  #
#  if ds.attrs[cpp].find('EW_PERIODIC')!= -1 and ds.attrs[cpp].find('NS_PERIODIC')!= -1:
#    coords={'x':{'center':'xi_rho',  'left':'xi_u'},
#            'y':{'center':'eta_rho', 'left':'eta_v'},
#            'z':{'center':'s_rho',   'outer':'s_w'}}
#    grid = Grid(ds,
#            coords=coords,
#            periodic=True)
#  else:
#    coords={'x':{'center':'xi_rho',  'inner':'xi_u'},
#            'y':{'center':'eta_rho', 'inner':'eta_v'},
#            'z':{'center':'s_rho',   'outer':'s_w'}}
#    grid = Grid(ds,
#            coords=coords,
#            boundary='extend')
#  #####################
#
#  name=str('d%s_%s' % (component, var))
#
#  if var='temp':
#    print('-->> TODO <<--')
#  elif var='u':
#    print('-->> TODO <<--')
#  elif var='v': 
#    print('-->> TODO <<--')
#  elif var='w':
#    if component='x':
#     
#    
#
#  if interp:
#    ds[name] = grid.interp(grid.diff(eval('ds.%s' % var)) , component)
#
#
#
#
#    return tmpout
