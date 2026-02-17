#!/usr/bin/env python

import numpy as np
import xarray as xr
from itertools import combinations


def not_in(ref_list, loc_list, agg=True):
    """
    Find elements of list ref_list not in loc_list, and aggregate if agg=True (default).
    Used below to construct labels of dimensions along which means are computed.
    """
    out_list = tuple(item for item in ref_list if item not in loc_list)
    if agg:
        out_list = ''.join(out_list)
    return out_list




def anova(da, dims=None):
    """
    Compute ANAlyse Of VAriance (ANOVA) across the given dimension dims.
    
    Compute global variance, variance associated to each dimension (i.e. main effect), and interactions across dimensions (up to 4th order).
    
    Input:
        - ds: xarray DataArray
        - dims: list of dimensions to perform ANOVA.
                If len(dims)==1, this is equivalent to standard variance, with a biased estimator (i.e. ddof=0). 
                If len(dims)==2, 2-way ANOVA, with up to 2nd order interaction terms
                If len(dims)==3, 3-way ANOVA, with up to 3rd order interaction terms
                If len(dims)==4, 4-way ANOVA, with up to 4th order interaction terms
                Higher ANOVA are not coded yet (08/11/2025).
    """
    
    #-- verify that da is a xarray dataArray, then convert into a dataset --
    if not isinstance(da, xr.DataArray):
        print("STOP: da is not a xr.DataArray. Please provide a xr.DataArray as input.")
        return
    else:
        ds = da.copy().to_dataset()
        ds = ds.rename({da.name: 'varin'})
        
    
    #-- estimate number of dimensions --
    ndim = len(dims)
    if ndim == 1:
        print("STOP: ANOVA along 1 dimension is equivalent to standard variance computation.")
        return
    elif ndim > 4:
        print("STOP: 5-way and higher ANOVA have not been coded yet :/.")
        return
    #- define labels for each dimension -
    labels = tuple(str(idim) for idim in range(1, ndim+1))
    
    #-------------------
    #-- compute means --
    #-------------------    
    for imean in range(1, ndim+1):
        #- get all possible combinaisons of dimensions -
        dims_comb = tuple(combinations(dims, imean))
        labs_comb = tuple(combinations(labels, imean))
        #- compute means for each combinaison -
        for icomb in range(len(dims_comb)):
            #- extract dimensions associated with the local combinaison -
            local_dims = dims_comb[icomb]
            #- ascribe a label for referencing -
            local_lab = ''.join(labs_comb[icomb])
            #- compute mean -
            print(f"Compute mean over dimension: {local_dims}")
            ds[eval(f"'m_{local_lab}'")] = da.mean(dim=local_dims)
            #- rename global mean -
            if imean == ndim:
                ds = ds.rename({f"m_{local_lab}": "m_glo"})
            
    #-----------------------
    #-- compute variances --
    #-----------------------
    #-- global variance --
    ds["var_glo"] = ((ds.varin-ds.m_glo)**2).mean(dim=dims)
    
    #-- first order, i.e. main effects --
    for iii in range(len(dims)):
        #- label of the mean over all but the considered dimension -
        avg_lab = not_in(labels, labels[iii])
        #- compute variance -
        ds[eval(f"'var_{labels[iii]}'")] = eval(f"( (ds.m_{avg_lab} - ds.m_glo)**2 ).mean(dim=dims[iii])")
        
    #-- second order --
    dims_comb = tuple(combinations(dims, 2))
    labs_comb = tuple(combinations(labels, 2))
    for icomb in range(len(dims_comb)):
        #- local dimension across which to compute 2nd order interaction terms -
        local_dims = dims_comb[icomb]
        local_labs = labs_comb[icomb]
        local_lab  = ''.join(local_labs)
        print(f"Compute 2nd order interaction terms associated with dimensions: {local_dims}")
        #- labels for the means over all but the considered dimensions -
        # 1/ start with the 2 dimensional mean
        avg_lab = not_in(labels, local_labs)
        if ndim == 2:
            cff = ds.varin
        else:
            cff = eval(f"ds.m_{avg_lab}")
        # 2/ substract 1 dimensional means
        for iii in range(2):
            avg_lab = not_in(labels, local_labs[iii])
            cff -= eval(f"ds.m_{avg_lab}")
        # 3/ add global mean
        cff += ds.m_glo
        # 4/ square, compute the mean and store
        ds[eval(f"'var_{local_lab}'")] = (cff**2).mean(dim=local_dims)

    
    #-- third order terms --
    if ndim > 2:
        dims_comb = tuple(combinations(dims, 3))
        labs_comb = tuple(combinations(labels, 3))
        for icomb in range(len(dims_comb)):
            #- local dimension across which to compute 3rd order interaction terms -
            local_dims = dims_comb[icomb]
            local_labs = labs_comb[icomb]
            local_lab  = ''.join(local_labs)
            print(f"Compute 3rd order interaction terms associated with dimensions: {local_dims}")
            # 1/ start with the 3 dimentional mean
            avg_lab   = not_in(labels, local_labs)
            if ndim == 3:
                cff = ds.varin
            else:
                cff = eval(f"ds.m_{avg_lab}")
            # 2/ substract 2 dimensional means
            dims_comb_2 = tuple(combinations(local_dims, 2))
            labs_comb_2 = tuple(combinations(local_labs, 2))
            for iicomb in range(len(dims_comb_2)):
                avg_lab = not_in(labels, labs_comb_2[iicomb])
                cff -= eval(f"ds.m_{avg_lab}")
            # 3/ add 1 dimensional means
            for iii in range(3):
                avg_lab = not_in(labels, local_labs[iii])
                cff += eval(f"ds.m_{avg_lab}")
            # 4/ substract global mean
            cff -= ds.m_glo
            # 5/ square, compute the mean and store
            ds[eval(f"'var_{local_lab}'")] = (cff**2).mean(dim=local_dims)
            
            
    #-- fourth order terms --
    if ndim > 3: 
        dims_comb = tuple(combinations(dims, 4))
        labs_comb = tuple(combinations(labels, 4))
        for icomb in range(len(dims_comb)):
            #- local dimension across which to compute 4rd order interaction terms -
            local_dims = dims_comb[icomb]
            local_labs = labs_comb[icomb]
            local_lab  = ''.join(local_labs)
            print(f"Compute 4th order interaction terms associated with dimensions: {local_dims}")
            # 1/ start with the 4 dimensional mean
            avg_lab = not_in(labels, local_labs)
            if ndim == 4:
                cff = ds.varin
            else:
                cff = eval(f"ds.m_{avg_lab}")
            # 2/ substract 3 dimensional means
            dims_comb_3 = tuple(combinations(local_dims, 3))
            labs_comb_3 = tuple(combinations(local_labs, 3))
            for iicomb in range(len(dims_comb_3)):
                avg_lab = not_in(labels, labs_comb_3[iicomb])
                cff -= eval(f"ds.m_{avg_lab}")
            # 3/ add 2 dimension means
            dims_comb_2 = tuple(combinations(local_dims, 2))
            labs_comb_2 = tuple(combinations(local_labs, 2))
            for iicomb in range(len(dims_comb_2)):
                avg_lab = not_in(labels, labs_comb_2[iicomb])
                cff += eval(f"ds.m_{avg_lab}")
            # 4/ substract 1 dimension means
            for iii in range(4):
                avg_lab = not_in(labels, local_labs[iii])
                cff -= eval(f"ds.m_{avg_lab}")
            # 5/ add global mean
            cff += ds.m_glo
            # 6/ square, compute mean and store
            ds[eval(f"'var_{local_lab}'")] = (cff**2).mean(dim=local_dims)
                
    return ds
