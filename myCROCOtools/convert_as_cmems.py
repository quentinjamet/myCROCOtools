#!/usr/bin/env python

import numpy as np
import xarray as xr
import datetime
import os
import glob
from scipy.interpolate import griddata



################################
def generate_dir_list(dir_in, start_date, end_date):
    months = []
    current_date = start_date
    
    while current_date <= end_date:
        months.append("%s/%s" % (dir_in, current_date.strftime("%Y/%m")) )  # Format: 'YYYY/MM'
        # move to first day of next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    #-- The end! --
    return months

################################
def creat_dataset(depth, lat, lon, nmem, nt, attrs_his):

    #-- get dimensions and mesh --
    [nz, ny, nx] = [len(depth), len(lat), len(lon)]
    [e2t, e1t]   = [lat[1]-lat[0], lon[0]-lon[1]]
    
    #-- create dataset --
    ds = xr.Dataset(
        data_vars=dict(
            zos=(["number", "time", "latitude", "longitude"], np.zeros([nmem, nt, ny, nx])),
            uo=(["number", "time", "depth", "latitude", "longitude"], np.zeros([nmem, nt, nz, ny, nx])),
            vo=(["number", "time", "depth", "latitude", "longitude"], np.zeros([nmem, nt, nz, ny, nx])),
            thetao=(["number", "time", "depth", "latitude", "longitude"], np.zeros([nmem, nt, nz, ny, nx])),
            so=(["number", "time", "depth", "latitude", "longitude"], np.zeros([nmem, nt, nz, ny, nx])),
        ),
        coords=dict(
            number=np.arange(nmem).astype('int16'),
            time=np.zeros(nt).astype('float'),
            depth=depth.astype('float'),
            longitude=lon.astype('float'),
            latitude=lat.astype('float'),
        ),
        attrs=dict(title='Hourly mean fields from Ensemble Global Ocean Physics Analysis ',
                   comments='Retrieve from Mercator Ocean International ftp',
                   history=[attrs_his, \
                            "%s -- convertion to CMEMS product type by Quentin Jamet (quentin.jamet@shom.fr)" % \
                            datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') ],
                   easting='longitude',
                   field_type='mean',
                   northing='latitude',
                   Conventions='CF-1.6',
               )
    )
    
    # variables attributes
    ds.number.attrs      = {'long_name': 'ensemble member numerical id',
                        'units': '1',
                        'standard_name': 'realization'}
    ds.depth.attrs       = {'axis': 'Z',
                         'long_name': 'Depth',
                         'positive': 'down',
                         'standard_name': 'depth',
                         'unit_long': 'Meters',
                         'units': 'm'}
    ds.longitude.attrs   = {'axis': 'X',
                        'long_name': 'Longitude',
                        'standard_name': 'longitude',
                        'unit_long': 'Degrees East',
                        'units': 'degrees_east',
                        'step': e1t,
                        'valid_max': lon.max(),
                        'valid_min': lon.min()}
    ds.latitude.attrs    = {'axis': 'Y',
                        'long_name': 'Latitude',
                        'standard_name': 'latitude',
                        'unit_long': 'Degrees North',
                        'units': 'degrees_north',
                        'step': e2t,
                        'valid_max': lat.max(),
                        'valid_min': lat.min()}
    ds.zos.attrs         = {'cell_methods': 'area: mean',
                       'long_name': 'Sea surface height',
                       'standard_name': 'sea_surface_height_above_geoid',
                       'unit_long': 'Meters',
                       'units': 'm'}
    ds.uo.attrs          = {'cell_methods': 'area: mean',
                        'long_name': 'Eastward velocity',
                        'standard_name': 'eastward_sea_water_velocity',
                        'unit_long': 'Meters per second',
                        'units': 'm s-1'}
    ds.vo.attrs          = {'cell_methods': 'area: mean',
                        'long_name': 'Northward velocity',
                        'standard_name': 'northward_sea_water_velocity',
                        'unit_long': 'Meters per second',
                        'units': 'm s-1'}
    ds.thetao.attrs      = {'cell_methods': 'area: mean',
                        'long_name': 'Temperature',
                        'standard_name': 'sea_water_potential_temperature',
                        'unit_long': 'Degrees Celsius',
                        'units': 'degrees_C'}
    ds.so.attrs          = {'cell_methods': 'area: mean',
                        'long_name': 'Salinity',
                        'standard_name': 'sea_water_salinity',
                        'unit_long': 'Practical Salinity Unit',
                        'units': '1e-3'}
    
    
    #-- The end! --
    return ds

################################
def convert_ens025(dir_in, dates=None, mydata=None, varList=None, fileOUT=None):
    '''
    Reorganize ORCA025 ensemble simulation from Mercator Ocean following CMEMS convention.
    Data have been retrieved on the native NEMO grid, with one file per 3D variables (SSH is in a file with other 2D variables (e.g. air-sea fluxes)),
    and one file per ensemble memeber and per daily averaged output.
    This script interpolates all variables on regular grid, lump ensemble members within the same netcdf file and rename some variables following CMEMS convention.
    
    Input:
        - dir_in: the directory where to load original data. 
                  Assumes data are organized as: ${dir_in}/yyyy/mm/memXXX/
                  and each variables are stores into a dedicated file in the format *thetao*YYYYMMDD-YYYYMMDD.nc
        - dates: dictionary with initial and final dates (i.e. [month, year]) to consider, for directory scanning.
                Should be in the format: dates = dict({"yr_ini": YYYY, "mth_ini": MM, "yr_end": YYYY, "mth_end": MM})
        - varList: List of variable names associated with temperature ('thetao'), salinity ('so'), 
                    zonal ('uo') and meridional ('vo') velocity, sea surface elevation ('zos'), 
                    time, longitude, latitude, x and y grid index.
                Should be in the format: 
                varList = dict({"thetao": "",
                                "so": "", 
                                "uo": "", 
                                "vo": "", 
                                "zos": "", 
                                "time": "", 
                                "longitude": "", 
                                "latitude": "",
                                "x": "", 
                                "y": "",
                                "z": "",
                                "x_u": "",
                                "y_u": "",
                                "z_u": "",
                                "x_v": "",
                                "y_v": "",
                                "z_v": "",})
    '''
    
    #-- check the type of data and associated list of variables --
    if varList is None:
        print("STOP -- Please provide the name of variables associated to the type of data to load.")
        print("STOP -- Should be in a dictionary format (see description).")
        return
    
    #-- check time period to consider, extract nane of associated sub-directories --
    if dates is None: 
        print("STOP -- Please provide dates in a dictionary.")
        return
    if (datetime.datetime(dates['yr_end'], dates['mth_end'], 1, 0, 0).toordinal() - datetime.datetime(dates['yr_ini'], dates['mth_ini'], 1, 0, 0).toordinal()) < 0:
        print("-- End date (%04.i/%02.i) is prior to start date (%04.i/%02.i) -- Please provide time increasing dates." % (dates['yr_end'], dates['mth_end'], dates['yr_ini'], dates['mth_ini']))
        return
    dir_list = generate_dir_list(dir_in, datetime.date(dates['yr_ini'], dates['mth_ini'], 1), datetime.date(dates['yr_end'], dates['mth_end'], 1))
    for idir in dir_list:
        if not os.path.isdir(idir):
            print("!!! Directory %s not found !!!" % idir )
            return
        
    #-- check output file name --
    if fileOUT is None:
        print("STOP -- Please provide a prefix for output file names")
        return
    
    #-- construct regular grid ; get spatial dimensions --
    tmpfileN = sorted( glob.glob("%s/mem*/*%s*" % (dir_list[0], varList['thetao']) ) )
    tmp = xr.open_dataset("%s" % tmpfileN[0])
    #
    [e2t, e1t] = [(tmp.nav_lat[1:, :]-tmp.nav_lat[:-1, :]).mean().data, \
                  (tmp.nav_lon[:, 1:]-tmp.nav_lon[:, :-1]).mean().data]
    longitude = np.arange(tmp.nav_lon.min(), tmp.nav_lon.max()+e1t, e1t)
    latitude  = np.arange(tmp.nav_lat.min(), tmp.nav_lat.max()+e2t, e2t)
    depth     = tmp.deptht.data
    #
    [nz, ny, nx] = [tmp.dims["deptht"], len(latitude), len(longitude)]
    
    #-- construct assiated land/ocean mask --
    print("-- Generate 3D land/ocean mask on regular grid --")
    msk = xr.where(tmp.thetao.isel(time_counter=0) > 0, 1., 0.)
    msk = msk.stack(yx=("y", "x"))
    msk_reg = np.zeros([nz, ny, nx])
    for kkk in range(nz):
        msk_reg[kkk, ...] = griddata((msk.nav_lon, msk.nav_lat), msk[kkk, :], \
                                     (longitude[None, :], latitude[:, None]), method='linear')
    msk_reg = np.where(msk_reg > 0.999, 1., np.nan)
    
    #-- attributes --
    attrs_his = tmp.attrs['history']
    
    #-------------------
    # Loop on each month
    #-------------------
    for idir in dir_list:
        print("-- Convert month: %s/" % idir.replace(dir_in, ""))
        
        #- get ensemble and time dims -
        nmem = len(glob.glob("%s/mem*" % idir))
        
        #- initiate xarray dataset -
        ds = creat_dataset(depth, latitude, longitude, nmem, 1, attrs_his)
    
        #----------------------------
        # Interpolate on regular grid
        #----------------------------
        
        #-- get number of time frame and file extension assiated with their dates --
        tmplist = sorted(glob.glob("%s/mem000/*%s*.nc" % (idir, varList['thetao'])))
        nfile   = len(tmplist)
        
        for ttt in range(nfile):
            ext = tmplist[ttt][-20:]
            print("time period: %s" % (ext[:-3]) )
            
            for imem in range(nmem):
                print("== member %03.i ==" % (imem) )
                
                #-- zos --
                #print("    - zos")
                # load 
                tmpda = eval("xr.open_mfdataset('%s/mem%03.i/*%s*%s').%s" % \
                             (idir, imem, varList['zos'], ext, varList['zos']) )
                # extract time info
                if imem == 0: 
                    ds["time"] = eval("tmpda.%s.data.astype('float')" % (varList['time']) )
                    ds.time.attrs = dict(time_origin=eval("tmpda.%s.attrs['time_origin']" % varList['time']))
                # prepare for interpoaltion (remove time dimension, stack on the hz, remove NaNs)
                tmpda = eval("tmpda.isel(%s=0).stack(yx=('%s', '%s')).dropna(dim='yx', how='any')" % \
                             (varList['time'], varList['y'], varList['x']))
                # interpolate
                tmpvar = eval("griddata((tmpda.%s, tmpda.%s), tmpda, (longitude[None, :], latitude[:, None]), method='linear')" % \
                              (varList['longitude'], varList['latitude']))
                # apply mask and write into the new dataset
                ds["zos"][imem, 0, ...] = (tmpvar * msk_reg[0, ...])
                
                #-- thetao --
                #print("    - thetao")
                for kkk in range(nz):
                    tmpda = eval("xr.open_mfdataset('%s/mem%03.i/*%s*%s').%s.isel(%s=0, %s=kkk).stack(yx=('%s', '%s')).dropna(dim='yx', how='any')" % \
                                ( idir, imem, varList['thetao'], ext, varList['thetao'], \
                                  varList['time'], varList['z'], varList['y'], varList['x']) )
                    if tmpda.size != 0:
                        tmpvar = eval("griddata((tmpda.%s, tmpda.%s), tmpda, (longitude[None, :], latitude[:, None]), method='linear')" % (varList['longitude'], varList['latitude']) ) 
                        ds["thetao"][imem, 0, kkk,  ...] = (tmpvar * msk_reg[kkk, ...])
                    else:
                        break
                        
                #-- so --
                #print("    - so")
                for kkk in range(nz):
                    tmpda = eval("xr.open_mfdataset('%s/mem%03.i/*%s*%s').%s.isel(%s=0, %s=kkk).stack(yx=('%s', '%s')).dropna(dim='yx', how='any')" % \
                                ( idir, imem, varList['so'], ext, varList['so'], \
                                  varList['time'], varList['z'], varList['y'], varList['x']) )
                    if tmpda.size != 0:
                        tmpvar = eval("griddata((tmpda.%s, tmpda.%s), tmpda, (longitude[None, :], latitude[:, None]), method='linear')" % (varList['longitude'], varList['latitude']) ) 
                        ds["so"][imem, 0, kkk,  ...] = (tmpvar * msk_reg[kkk, ...])
                    else:
                        break
                        
                #-- uo --
                #print("    - uo")
                for kkk in range(nz):
                    tmpda = eval("xr.open_mfdataset('%s/mem%03.i/*%s*%s').%s.isel(%s=0, %s=kkk).stack(yx=('%s', '%s')).dropna(dim='yx', how='any')" % \
                                ( idir, imem, varList['uo'], ext, varList['uo'], \
                                  varList['time'], varList['z_u'], varList['y_u'], varList['x_u']) )
                    if tmpda.size != 0:
                        tmpvar = eval("griddata((tmpda.%s, tmpda.%s), tmpda, (longitude[None, :], latitude[:, None]), method='linear')" % (varList['longitude'], varList['latitude']) ) 
                        ds["uo"][imem, 0, kkk,  ...] = (tmpvar * msk_reg[kkk, ...])
                    else:
                        break
                        
                #-- vo --
                #print("    - vo")
                for kkk in range(nz):
                    tmpda = eval("xr.open_mfdataset('%s/mem%03.i/*%s*%s').%s.isel(%s=0, %s=kkk).stack(yx=('%s', '%s')).dropna(dim='yx', how='any')" % \
                                ( idir, imem, varList['vo'], ext, varList['vo'], \
                                  varList['time'], varList['z_v'], varList['y_v'], varList['x_v']) )
                    if tmpda.size != 0:
                        tmpvar = eval("griddata((tmpda.%s, tmpda.%s), tmpda, (longitude[None, :], latitude[:, None]), method='linear')" % (varList['longitude'], varList['latitude']) ) 
                        ds["vo"][imem, 0, kkk,  ...] = (tmpvar * msk_reg[kkk, ...])
                    else:
                        break
        
            #-- write to netcdf --
            print("(%s) -- write to netcdf: %s/%s_%s" % \
                  (datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), idir, fileOUT, ext))
            #print(str('%s/ens025-reana-daily_%s' % (idir, ext) ))
            ds.to_netcdf(path=str('%s/%s_%s' % (idir, fileOUT, ext) ), engine='netcdf4', compute=False)
            
    #-- The end! --
    return ds
