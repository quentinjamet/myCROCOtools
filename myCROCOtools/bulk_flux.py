import numpy as np
from   xgcm import Grid
from   datetime import datetime
import sys
import xarray as xr

class BulkFluxCOARE:
    """
    Vectorized Python translation of CROCO's bulk_flux module
    Computes turbulent and radiative air-sea fluxes using the COARE algorithm
    """
    
    def __init__(self, cfb=False, sto=False):
        # Physical constants
        self.g          = 9.81                                # Gravitational acceleration [m/s^2]
        self.vonKar     = 0.41                                # von Karman constant
        self.CtoK       = 273.16                              # Celsius to Kelvin conversion
        self.SigmaSB    = 5.6697e-8                           # Stefan-Boltzmann constant
        self.blk_Rgas   = 287.0596736665907                   # Gas constant for dry air
        self.blk_Cpa    = 1004.708857833067                   # Specific heat of air at constant pressure
        self.blk_Rgas   = 287.0596736665907                   # gas constant for dry air  [J/(kg K)]
        self.blk_Rvap   = 461.5249933083879                   # gas constant for water vapor [J/(kg K)]
        self.blk_Cpa    = 1004.708857833067
        self.cpvir      = self.blk_Rvap/self.blk_Rgas - 1.    # Constant for virtual temperature
        self.emiss_lw   = 0.985                               # Ocean longwave emissivity
        self.ip00       = 1e-5                                # Inverse reference pressure [1/Pa]
        self.rdocpd     = self.blk_Rgas/self.blk_Cpa          # R_dry / cp_dry
        self.r_gas      = 8.314510                            # Universal gas constant
        self.mm_dryair  = 28.9644e-3                          # Molar mass of dry air
        self.mm_water   = 18.0153e-3                          # Molar mass of water vapor
        self.MvoMa      = self.mm_water/self.mm_dryair        # Ratio of molar masses vapor/dry air
        self.rho0       = 1025.0                              # Boussinesque Approximation Mean density [kg/m^3]
        self.rho0i      = 1.0 / self.rho0
        self.cp         = 3985.0                              # Heat capacity at constant pressure [J/K]
        self.cpi        = 1.0 / self.cp
        self.r3         = 1.0/3.0
        self.eps        = 1e-20
        self.Cd         = 1.2e-3                              # Canonical value for air-sea momentum exchange coef [nd]
        
        # Height parameters
        self.blk_ZW = 10.0         # Wind height [m]
        self.blk_ZT = 2.0          # Temperature/humidity height [m]
        self.blk_ZToZW = self.blk_ZT / self.blk_ZW
        self.Log10oLogZw = np.log(10.0 * 10000.) / np.log(self.blk_ZW * 10000.)
        
        # Convection parameters
        self.blk_beta = 1.25       # Gustiness parameter
        self.blk_Zabl = 600.0      # Atmospheric boundary layer height
        
        # Constants for stability functions
        self.sqr3 = np.sqrt(3.0)
        self.pis2 = np.pi / 2.0
        self.pis2osqr3 = self.pis2 / self.sqr3
        
        # Iteration parameters
        self.IterFl = 3
        
        # current feedback parameters
        self.cfb = cfb                   #
        if cfb:
            self.mb = 2                      # to compute Wstar associated with both absolute and relative wind formulation
            self.swparam = 0.3               # wind correction: Ua-(1-sw)*Uo
            self.cfb_slope=-0.0029           # wind-stress correction using wind speed:  rho0*sustr + s_tau*Uo
            self.cfb_offset=0.008            #            s_tau = cfb_slope * wspd + cfb_offset [N.m^-3.s]
        else:
            self.mb = 1

        # stochastic perturbation
        self.sto = sto
        if sto:
            self.ave     = 0.0               # mean of the distribution
            self.std     = 0.1               # standard deviation
            self.distrib = 'lognormal'       # type of distribution ('normal', 'lognormal')
            self.npasses = 20                # number of passes of Laplacian filter
            self.n_sto   = 10                # number of ensemble

    def timemonitor(self):
        print(f"--- {(datetime.utcnow()-self.t0).total_seconds()} sec --")
        self.t0 = datetime.utcnow()

    def test_xarray(self, var, name):
        if isinstance(var, xr.DataArray):
            print(f"--- TEST --- {name} is an xr.DataArray ---")
            print(f"--- TEST ---{name}.dims: {var.dims}")
        else:
            print(f"--- TEST --- {name} is NOT an xr.DataArray ---")
            print(f"--- TEST ---{name}.shape: {var.shape}")

        
    def make_xgrid(self, ds):
        """
        Construct xgcm Grid for interpolation.
        Remove extra ghost points if needed for xgcm to work properly,
        i.e. 'inner' could have been used with xi_u and eta_v, 
        but this follows the 'right' staggerging convention and CROCO is written with the 'left' convention
        (cf https://xgcm.readthedocs.io/en/latest/grids.html).
        """
        #-- check dataset and remove ghost points if needed --
        if ds.dims['xi_rho'] != ds.dims['xi_u']:
            print("rho-point and u-point do not have the same dimension. "
                  "Assumes presence of ghost points (following CROCO grid convention), and remove them.")
            ds = ds.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1), eta_v=slice(0, -1), xi_u=slice(0, -1), eta_psi=slice(0, -1), xi_psi=slice(0, -1))
        #-- define xgcm grid --
        coords={'x':{'center':'xi_rho',  'left':'xi_u'},
                'y':{'center':'eta_rho', 'left':'eta_v'}}
        self.xgrid = Grid(ds,
                          coords=coords,
                          boundary="extend")
        #-- check grid definition --
        # TBD


    def stogen(self, da_in, mask):
        """
        Random number generatore, in the spirit of  STOGEN but much lighter in capabilities (and performance likley).
        da_in is used as reference to construct stochatic fields with appropriate dimension.
        Can generate normal and lognormal distribution (defined in __init__()) and account for land-sea mask.
        """
        
        dims = ["number_sto"] + list(da_in.dims)
        shape = (self.n_sto,) + tuple(da_in.sizes[d] for d in da_in.dims)
        
        if self.distrib == 'normal':
            da_sto = xr.DataArray(
                np.random.normal(self.ave, self.std, shape),
                dims=dims,
            ) * mask
        elif self.distrib == 'lognormal':
            da_sto = xr.DataArray(
                np.random.lognormal(self.ave, self.std, shape),
                dims=dims,
            ) * mask
        else:
            sys.exit(f"(in stogen) Bad type of requested distribution. Available are: 'normal', 'lognormal'. Provided: {self.distrib}.")

        #-- apply npasses of Laplacian filter --
        for ipasse in range(self.npasses):
            cff = (self.xgrid.diff(self.xgrid.diff(da_sto, 'x'), 'x') + 
                   self.xgrid.diff(self.xgrid.diff(da_sto, 'y'), 'y'))
            da_sto = da_sto + cff * mask

        #-- restore to original std (without counting land points) --
        da_sto = da_sto * self.std / da_sto.where(da_sto != 0).stack(xy=['eta_rho', 'xi_rho']).std(dim='xy')
        
        return da_sto


    def spec_hum(self, RH, psfc, TairC):
        """
        Computes specific humidity from relative humidity
        """
        # If RH < 2, it's a fraction, otherwise it's in g/kg -- random selection into xarray multidimensional table RH
        cff2 = 0.0
        iii = 0
        while cff2 == 0.0 or np.isnan(cff2):
            cff2 = RH.to_numpy().flat[np.random.choice(RH.size, size=1, replace=False)]
            iii += 1
            if iii > 100:
                sys.exit("(in spec_hum) Can't find non-NaN values in RH. Specific humidiy not computed.")
        
        if cff2 < 2.0:
            # RH as fraction
            # Compute saturated vapor pressure (Teten's formula)
            cff = (1.0007 + 3.46e-6 * 0.01 * psfc) * 6.1121 * \
                  np.exp(17.502 * TairC / (240.97 + TairC))
            cff_frac = cff * RH
            spec_hum = self.MvoMa * (cff_frac / (psfc * 0.01 - 0.378 * cff_frac))
        else:
            # RH in g/kg 
            spec_hum = 0.001 * RH
        
        return spec_hum

    def qsat(self, TairK, patm, coeff=1.0):
        """Computes saturation humidity"""
        # Parameters for liquid water
        alpw = 60.2227554
        betaw = 6822.40088
        gamw = 5.13926744
        
        # Parameters for ice
        alpi = 32.62117980819471
        betai = 6295.421338904806
        gami = 0.5631331575423155
        
        # Mask for freezing temperature
        ice_mask = TairK <= self.CtoK
        
        # Saturated vapor pressure
        psat_ice = np.exp(alpi - betai/TairK - gami*np.log(TairK))
        psat_water = np.exp(alpw - betaw/TairK - gamw*np.log(TairK))
        
        psat = xr.where(ice_mask, psat_ice, psat_water) * coeff
        
        # Saturation humidity
        return (self.MvoMa * psat) / (patm + (self.MvoMa - 1.0) * psat)

    def exner_patm_from_tairabs(self, q, tairabs, z, psfc, Niter=3):
        """
        Computes Exner function and atmospheric pressure
        """
        pair = psfc.copy()
        
        for _ in range(Niter):
            q_sat = self.qsat(tairabs, pair, 1.0)
            xm = self.mm_dryair + (q/q_sat) * (self.mm_water - self.mm_dryair)
            pair = psfc * np.exp(-self.g * xm * z / (self.r_gas * tairabs))
        
        iexn = (pair * self.ip00) ** (-self.rdocpd)
        return iexn, pair

    def air_visc(self, TairC):
        """Molecular viscosity of air as a function of temperature"""
        c0, c1, c2, c3 = 1.326e-5, 6.542e-3, 8.301e-6, 4.84e-9
        TairC2 = TairC * TairC
        return c0 * (1.0 + c1*TairC + c2*TairC2 - c3*TairC2*TairC)

    def bulk_psiu_coare(self, ZoL):
        """COARE stability function for velocity"""
        unstable = ZoL <= 0.0
    
        tmpZoL = xr.where(unstable, ZoL, 0.0)
    
        chik = (1.0 - 15.0 * tmpZoL) ** 0.25
        psik = (2.0 * np.log(0.5 * (1.0 + chik)) +
                np.log(0.5 * (1.0 + chik**2)) -
                2.0 * np.arctan(chik) + self.pis2)
    
        chic = (1.0 - 10.15 * tmpZoL) ** self.r3
        psic = (1.5 * np.log(self.r3 * (chic**2 + chic + 1.0)) -
                self.sqr3 * np.arctan((2.0*chic + 1.0)/self.sqr3) +
                2.0 * self.pis2osqr3)
    
        psi_unstable = psic + (psik - psic) / (1.0 + ZoL**2)
    
        tmpZoL = xr.where(~unstable, ZoL, 0.0)
        chic_stable = -(0.35 * tmpZoL).clip(max=50.0)
        psi_stable = -((1.0 + tmpZoL) + 0.6667 * (tmpZoL - 14.28) *
                      np.exp(chic_stable) + 8.525)
    
        return xr.where(unstable, psi_unstable, psi_stable)
    
    
    def bulk_psit_coare(self, ZoL):
        """COARE stability function for tracers"""
        unstable = ZoL < 0.0
    
        tmpZoL = xr.where(unstable, ZoL, 0.0)
    
        chik = (1.0 - 15.0 * tmpZoL) ** 0.25
        psik = 2.0 * np.log(0.5 * (1.0 + chik**2))
    
        chic = (1.0 - 34.15 * tmpZoL) ** self.r3
        psic = (1.5 * np.log((chic**2 + chic + 1.0) * self.r3) -
                self.sqr3 * np.arctan((2.0*chic + 1.0)/self.sqr3) +
                2.0 * self.pis2osqr3)
    
        psi_unstable = psic + (psik - psic) / (1.0 + ZoL**2)
    
        tmpZoL = xr.where(~unstable, ZoL, 0.0)
        chic_stable = -(0.35 * tmpZoL).clip(max=50.0)
        psi_stable = -((1.0 + 2.0*tmpZoL/3.0)**1.5 +
                      0.6667 * (tmpZoL - 14.28) * np.exp(chic_stable) + 8.525)
    
        return xr.where(unstable, psi_unstable, psi_stable)

    def comp_wspd(self, ds_atm, ds_ocn):
        """
        Compute wind speed from atmospheric forcing file (assuming wind components both at rho-points),
        including current feedback if self.cfb=True. 
        In the latter case (i.e. self.cfb=True), ocean surface currents are first interpolated at rho-points.
        """
        #- compute wind speed -
        wspd = np.sqrt(ds_atm.U10M**2 + ds_atm.V10M**2)
        if self.cfb:
            cff      = 1.-self.swparam                  # current-wind coupling parameter: Ua => Ua-(1-sw)Uo
            wspd_cfb = np.sqrt( (ds_atm.U10M - cff*self.xgrid.interp(ds_ocn.u, 'x')*ds_ocn.mask_rho )**2
                               +(ds_atm.V10M - cff*self.xgrid.interp(ds_ocn.v, 'y')*ds_ocn.mask_rho )**2
                             )
        else:
            wspd_cfb = xr.zeros_like(wspd)

        return wspd, wspd_cfb
    
    def bulk_flux(self, ds_atm, ds_ocn):
        """
        Recompute CROCO air-sea fluxes based on COARE3 bulk algorithm.
        
        Parameters:
        -----------
        ds_atm: xarray dataset with regular atmospheric focing fields
            t_sea : array - Sea surface temperature (°C)
            tair : array - Air temperature (°C)  
            rhum : array - Relative humidity (fraction or g/kg)
            uwnd, vwnd : array - Wind components (m/s)
            patm2d : array - Atmospheric pressure (Pa)
            radlw : array - Downward longwave radiation (W/m²)
            srflx : array - Shortwave radiation (W/m²)
            prate : array - Precipitation (m/s)
        ds_ocn: xarray dataset with (surface) CROCO model outputs. 
                SST and surface currents are used respectively for:
                  - temperature scaling parameters (Tstar), stability, latent heat and freshwater fluxes, and long wave radiations
                  - current feedbacks (if cfb=True) estimates.
        
        Returns:
        --------
        Xarray dataset with with fluxes and diagnostic variables.
            - rho0*[sustr, svstr]

        Notes:
        ------
            - sustr, svstr are devided by rho0 at compute time (bulk_flux.F), 
              and multiplied by rho0 at output (wrt_his.F).
        """
        self.t00 = datetime.utcnow() 
        self.t0  = datetime.utcnow() 

        #-- check and adjust dimensions --
        # both have ensemble dimension ; rename and add them both to the other
        if 'number' in ds_ocn.dims and 'number' in ds_atm.dims:
            ds_atm = ds_atm.rename({'number': 'number_atm'})
            ds_ocn = ds_ocn.rename({'number': 'number_ocn'})
            ds_atm = ds_atm.expand_dims(dim={"number_ocn": ds_ocn.dims["number_ocn"]}, axis=1)
            ds_ocn = ds_ocn.expand_dims(dim={"number_atm": ds_atm.dims["number_atm"]}, axis=0)
        # only ocean dataset has ensemble dimension ; add the dimension to atmospheric dataset
        if 'number' in ds_ocn.dims and 'number' not in ds_atm.dims:
            ds_atm = ds_atm.expand_dims(dim={"number": ds_ocn.dims["number"]}, axis=0)
        # only atmospheric dataset has ensemble dimension ; add the dimension to ocean dataset
        if 'number' in ds_atm.dims and 'number' not in ds_ocn.dims:
            ds_ocn = ds_ocn.expand_dims(dim={"number": ds_atm.dims["number"]}, axis=0)

        #-- construct xgcm grid --
        print("Construct xgcm grid")
        self.make_xgrid(ds_ocn)
        #self.timemonitor()
        
        #-- extract variables from xarray dataset (to be changed) --
        t_sea = ds_ocn.temp
        mask  = ds_ocn.mask_rho
        tair  = ds_atm.T2M
        rhum  = ds_atm.R
        uwnd  = self.xgrid.interp(ds_atm.U10M, 'x')
        vwnd  = self.xgrid.interp(ds_atm.V10M, 'y')
        [wspd, wspd_cfb] = self.comp_wspd(ds_atm, ds_ocn)
        patm2d = ds_atm.MSL
        
        # generate stochastic 2D field, if needed
        if self.sto:
            print(f"Generate {self.n_sto} stochastic fields")
            sto2d = self.stogen(tair, mask)
            t_sea = t_sea * (1.0 + sto2d)
            print(f"t_sea.dims: {t_sea.dims}")
        
        # Basic atmospheric variables
        psurf = patm2d
        iexns = (psurf * self.ip00) ** (-self.rdocpd)
        
        wspd0 = np.maximum(wspd, 0.1 * np.minimum(10.0, self.blk_ZW))
        wspd0_cfb = np.maximum(wspd_cfb, 0.1 * np.minimum(10.0, self.blk_ZW))
 
        TairC = tair
        TairK = TairC + self.CtoK
        TseaC = t_sea
        TseaK = TseaC + self.CtoK
        
        # Specific humidity
        print("Compute specific humidity")
        Q = self.spec_hum(rhum, psurf, TairC)
        #self.timemonitor()
        
        # Atmospheric Exner function
        print("Compute Exner function")
        iexna, patm = self.exner_patm_from_tairabs(Q, TairK, self.blk_ZT, psurf)
        #self.timemonitor()
        
        # Air density
        print("Compute air density")
        rhoAir = patm * (1.0 + Q) / (self.blk_Rgas * TairK * (1.0 + self.MvoMa * Q))
        #self.timemonitor()
        
        # Surface saturation humidity
        print("Compute humidity at saturation")
        Qsea = self.qsat(TseaK, psurf, 0.98)
        #self.timemonitor()
        
        # Air-sea gradients
        print("Compute air-sea gradients")
        delW = xr.concat([
            np.sqrt(wspd0**2 + 0.25),
            np.sqrt(wspd0_cfb**2 + 0.25)
        ], dim="component")
        delQ = Q - Qsea
        #self.timemonitor()
        #self.test_xarray(delW, "delW")
        
        cff = self.CtoK * (iexna - iexns)
        delT = TairC * iexna - TseaC * iexns + cff
        
        # === COARE ALGORITHM ===
        
        # Initial estimate
        print("==== First guess ====")
        #Wstar = np.zeros_like(delW)
        Wstar = 0.035 * delW * self.Log10oLogZw
        
        VisAir = self.air_visc(TairC)
        charn = xr.full_like(TairC, 0.011)
        Ch10 = 0.00115
        Ribcu = -self.blk_ZW / (self.blk_Zabl * 0.004 * self.blk_beta**3)

        
        # Initial roughness length scale calculation
        Wstar_list = []
        for m in range(self.mb):
            Wstar_m = Wstar.isel(component=m)

            iZo10 = self.g * Wstar_m / (charn * Wstar_m**3 + 0.11 * self.g * VisAir)
            iZoT10 = 0.1 * np.exp(self.vonKar**2 / (Ch10 * np.log(10.0 * iZo10)))
            
            CC = (np.log(self.blk_ZW * iZo10)**2) / np.log(self.blk_ZT * iZoT10)
            Ri = (self.g * self.blk_ZW * (delT + self.cpvir * TairK * delQ) / 
                  (TairK * delW.isel(component=m)**2))
            
            # Stability parameter
            unstable = Ri < 0.0
            ZoLu_unstable = CC * Ri / (1.0 + Ri / Ribcu)
            ZoLu_stable = CC * Ri / (1.0 + 3.0 * Ri / CC)
            ZoLu = xr.DataArray(
                np.where(unstable.values, ZoLu_unstable.values, ZoLu_stable.values),
                dims=Ri.dims,
                coords=Ri.coords
            )
            
            psi_u = self.bulk_psiu_coare(ZoLu)
            logus10 = np.log(self.blk_ZW * iZo10)
            Wstar_list.append(delW.isel(component=m) * self.vonKar / (logus10 - psi_u))
       
        Wstar = xr.concat(Wstar_list, dim="component")
 
        ZoLt = ZoLu * self.blk_ZToZW
        psi_t = self.bulk_psit_coare(ZoLt)
        logts10 = np.log(self.blk_ZT * iZoT10)
        cff_t = self.vonKar / (logts10 - psi_t)
        Tstar = delT * cff_t
        Qstar = delQ * cff_t

        # Charnock coefficient as a function of wind
        print("Compute Charnock coef.")
        w0 = delW.isel(component=0)
        charn = xr.DataArray(
            np.where(w0.values > 18.0, 0.018,
            np.where(w0.values > 10.0,
                     0.011 + 0.125 * (0.018 - 0.011) * (w0.values - 10.0),
                     0.011)),
            dims=w0.dims, coords=w0.coords
        )

        self.timemonitor()
        
        # === ITERATIVE LOOP ===
        print("==== Iterative estimates ====")
        for iteration in range(self.IterFl):
            print(str(r"   (iter = %i/%i )" %(iteration, self.IterFl-1)))
            Wstar_list = []
            for m in range(self.mb):
                Wstar_m = Wstar.isel(component=m)

                # Roughness length
                iZoW = self.g * Wstar_m / (charn * Wstar_m**3 + 0.11 * self.g * VisAir)
                
                # Thermal roughness length
                Rr = Wstar_m / (iZoW * VisAir)
                iZoT = np.maximum(8695.65, 18181.8 * ( np.where(Rr > 0., Rr, 0.0)**0.6))
                
                # Monin-Obukhov stability parameter
                ZoLu = (self.vonKar * self.g * self.blk_ZW * 
                       (Tstar * (1.0 + self.cpvir * Q) + self.cpvir * TairK * Qstar) /
                       (TairK * Wstar_m**2 * (1.0 + self.cpvir * Q) + self.eps))

                # Stability functions
                psi_u = self.bulk_psiu_coare(ZoLu)
                logus10 = np.log(self.blk_ZW * iZoW)
                Wstar_list.append(delW.isel(component=m) * self.vonKar / (logus10 - psi_u))
            
            Wstar = xr.concat(Wstar_list, dim="component")
 
            ZoLt = ZoLu * self.blk_ZToZW  # Use the last calculated ZoLu
            psi_t = self.bulk_psit_coare(ZoLt)
            
            # Monin-Obukhov scales
            logts10 = np.log(self.blk_ZT * iZoT)
            cff_t = self.vonKar / (logts10 - psi_t)
            Tstar = delT * cff_t
            Qstar = delQ * cff_t
           
            # Gustiness (free convection)
            delW_list = []
            wspd_list = [wspd0, wspd0_cfb]  # indexé par m
            for m in range(self.mb):
                Bf = -self.g / TairK * Wstar.isel(component=m) * (Tstar + self.cpvir * TairK * Qstar)
                cff_gust = xr.where(Bf > 0.0,
                                    self.blk_beta * (xr.where(Bf > 0., Bf, 0.0) * self.blk_Zabl) ** self.r3,
                                    0.2)
                delW_list.append(np.sqrt(wspd_list[m]**2 + cff_gust**2))
            
            delW = xr.concat(delW_list, dim="component")
 
            self.timemonitor()
        
        # Transfer coefficients
        print("Transfer coefficient")
        aer  = rhoAir * delW.isel(component=0)
        Cd   = (Wstar.isel(component=0) / delW.isel(component=0))**2
        Cd_cfb = (Wstar.isel(component=1) / delW.isel(component=1))**2

        # wind stress (@ rho-pts)
        sustr = Cd*aer*ds_atm.U10M
        svstr = Cd*aer*ds_atm.V10M
        sustr_cstCd = self.Cd*aer*ds_atm.U10M
        svstr_cstCd = self.Cd*aer*ds_atm.V10M


        if self.cfb:
            # store estimates of absolute wind stress
            sustr0 = sustr.copy()
            svstr0 = svstr.copy()
            # update with current feedback (stress-correction approach)
            stau = self.cfb_slope * delW.isel(component=0) + self.cfb_offset
            sustr += stau * self.xgrid.interp(ds_ocn.u, 'x')
            svstr += stau * self.xgrid.interp(ds_ocn.v, 'y')
            sustr_cstCd += stau * self.xgrid.interp(ds_ocn.u, 'x')
            svstr_cstCd += stau * self.xgrid.interp(ds_ocn.v, 'y')
        
        # output 
        data_vars = {
            "cd"    : Cd,
            "sustr" : sustr,
            "svstr" : svstr,
        }

        ds_vars = {}
        for name, var in data_vars.items():
            ds_vars[name] = var.compute()
        
        ds_out = xr.Dataset(
            ds_vars,
            coords=ds_atm.coords,
            attrs=dict(description="Recomputed AO transfer coefficients, turbulent scales and associated fluxes, possibly stochastically perturbed"),
        )
        
        if self.sto:
            ds_out["sto2d"] = sto2d

        # ds_out = xr.Dataset(
        #     data_vars=dict(
        #         zolu        = (dims, ZoLu.data),
        #         psi_u       = (dims, psi_u.data),
        #         charn       = (dims, charn.data),
        #         delw        = (dims, delW[0, ::].data),
        #         wstar       = (dims, Wstar[0, ::].data),
        #         tstar       = (dims, Tstar.data),
        #         qstar       = (dims, Qstar.data),
        #         cd          = (dims, Cd),
        #         cd_cfb      = (dims, Cd_cfb),
        #         sustr0      = (dims, sustr0.data),
        #         svstr0      = (dims, svstr0.data),
        #         sustr       = (dims, sustr.data),
        #         svstr       = (dims, svstr.data),
        #         sustr_cstCd = (dims, sustr_cstCd.data),
        #         svstr_cstCd = (dims, svstr_cstCd.data),
        #         sto2d       = (dims_sto, sto2d.data)
        #     ),
        #     coords=ds_atm.coords,
        #     attrs=dict(description="Recomputed AO transfer coefficients, turbulent scales and associated fluxes, possibly stochastically perturbed"),
        # )
        #if self.cfb:
        #    ds_out["stau"] = (ds_ocn.temp.dims, stau)

        
        print(" DONE -- time: ", (datetime.utcnow()-self.t00).total_seconds(), ' sec')
        
        return ds_out

#         # === FLUX COMPUTATION ===
        
#         # Sensible heat flux (W/m²)
#         hfsen = -self.blk_Cpa * rhoAir * Wstar[1] * Tstar
        
#         # Latent heat of vaporization (J/kg)
#         Hlv = (2.5008 - 0.0023719 * TseaC) * 1.0e6
        
#         # Latent heat flux (W/m²)
#         hflat = -Hlv * rhoAir * Wstar[1] * Qstar
        
#         # Net longwave radiation (W/m²)
#         hflw = (radlw - self.emiss_lw * self.rho0i * self.cpi * 
#                 self.SigmaSB * TseaK**4)
        
#         # Webb correction for latent flux
#         upvel = (-1.61 * Wstar[1] * Qstar - 
#                 (1.0 + 1.61 * Q) * Wstar[1] * Tstar / TairK)
#         hflat = hflat + rhoAir * Hlv * upvel * Q
        
#         # Conversion to oceanic fluxes
#         hflat = -hflat * self.rho0i * self.cpi
#         hfsen = -hfsen * self.rho0i * self.cpi
        
#         # Total heat flux
#         stflx_temp = srflx + hflw + hflat + hfsen
        
#         # Salt flux (evaporation - precipitation)
#         evap = -cp * hflat / Hlv
#         stflx_salt = (evap - prate) * t_sea
