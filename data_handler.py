import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load pre-built data classes
from utils import Utils as utls

# all data can be accessed at
# https://drive.google.com/drive/folders/1vnZtue8enMPdxsyXcBxdoVPEc1BGGF6W?usp=sharing

data_folder = '../data/'
# GRACE DATA
# source: http://www2.csr.utexas.edu/grace/RL06_mascons.html
FNAME_DATA_CSR = data_folder + 'GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc'
FNAME_MASK_CSR = data_folder + 'GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_OceanMask.nc'
FNAME_DATA_JPL = data_folder + 'GRACE/data/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06_V2/GRCTellus.JPL.200204_202205.GLO.RL06M.MSCNv02CRI.nc'
FNAME_MASK_JPL = data_folder + 'GRACE/LAND_MASK.CRI.nc'
FNAME_DATA_GAD = data_folder + 'GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_GAD-component_v02.nc'
FNAME_MASK_GAD = data_folder + 'GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_OceanMask.nc'

# INDICES
FNAME_SAM = data_folder + 'SAM/SAM.txt'
FNAME_AAO = data_folder + 'SAM/AO.txt'
FNAME_IPO = data_folder + 'IPO/ipo.nc'

# DOTA DATA
# source: O. Dragomir et. al. 2021
FNAME_EGM = data_folder + 'DOT/dot_all_30bmedian_egm08.nc'
FNAME_GOCO = data_folder + 'DOT/dot_all_30bmedian_goco05c_sig3.nc'
FNAME_EIG = data_folder + 'DOT/dot_all_30bmedian_eigen6s4v2_sig3.nc'

# SLA DATA source: aviso
FNAME_SLA = data_folder + 'AVISO/dt_antarctic_multimission_sea_level_uv_20130401_20190731.nc'

# ERA5 DATA
# source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
FNAME_ERA = data_folder + 'ERA5/adaptor.mars.internal-1638887045.5747485-31674-12-a0391d02-7a74-4339-bb51-ea9d5feee59a.nc'
FNAME_ERA_GLOBAL = data_folder + 'ERA5/adaptor.mars.internal-1646140903.8330684-30797-17-d0d3e080-97c3-4464-a8b7-ae01799cbcdd.nc'

# GPHA
FNAME_PROFILES_DF = data_folder + 'GPHA/profiles.csv'
FNAME_PROFILES_GRIDDED = data_folder + 'GPHA/gridded_gpha.nc'

# BATHYMETRY DATA
FNAME_GEBCO = data_folder + 'GEBCO/gebco_2021_sub_ice_n-50.0_s-90.0_w-180.0_e180.0.nc'

# PROCESSED DATA
PROCESSED_FOLDER =  data_folder + 'PROCESSED/'

# Area of analysis
LAT_MIN = -81.75 #from dota
LAT_MAX = -50.25 #-50.25 #from dota

# Other constants
RHO = 1028 #kgm-3
G = 9.81 #ms-2


class GEBCO():
    def __init__(self,coarsen_factor=40):
        ds = xr.open_dataset(FNAME_GEBCO)
        ds = ds.rename({'lat':'latitude','lon':'longitude'})
        ds = ds.coarsen(latitude=coarsen_factor,longitude=coarsen_factor).mean()

        self.ds = ds

class Index():
    def __init__(self, type='SAM', start=None, end=None):
        """
        type = 'SAM' or 'AAO'
        """
        def open_txt(fname):
            data = np.genfromtxt(fname,skip_header=1,skip_footer=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
            years = np.repeat(np.genfromtxt(fname,skip_header=1,skip_footer=1,usecols=(0),dtype='int'),12)
            months = np.tile(np.arange(1,13),len(data))
            time = [pd.Timestamp(y,m,1) for y,m in zip (years,months)]
            data = data.reshape(np.size(data))
            return xr.DataArray(data,coords={'time': time})
        
        if type=='SAM':
            da = open_txt(FNAME_SAM)
        elif type=='AAO':
            da = open_txt(FNAME_AAO)
        elif type=='IPO':
            da = xr.open_dataarray(FNAME_IPO)
        else:
            print('Invalid index type')
            return None
        
        start = start or da.time[0]
        end = end or da.time[-1]

        self.da = da.sel(time=slice(start,end))

class DOT():
    # 
    def __init__(self, ver='goco',start=None,end=None):
        """ 
        ver = 'egm', 'goco' or 'eigen'
        """
        ds = self.__load(ver)
        
        # convert to cm
        dot_cm = ds.dot * 100
        
        # reorganise dims
        dot_cm = dot_cm.transpose("time","latitude","longitude")

        # assign to dataset with info
        ds['dot'] = dot_cm.assign_attrs({'description':'dynamic ocean topography (original)','units':'cm'})
        
        if (start is not None) and (end is not None):
            print('cropping DOT to {0} to {1}'.format(start,end))
            ds = ds.sel(time = slice(start,end))
            
        # mean over whole dataset timespan to compare with oanas
        mdt = ds.dot.mean('time',skipna=True)
        ds['mdt'] = mdt.assign_attrs({'description':'mean dynamic topography','units':'cm'})
        
        # anomalies wrt grid square mean
        dota = ds.dot - ds.mdt
        ds['dota'] = dota.assign_attrs({'description': 'dot anomaly wrt time mean per grid square', 'units': 'cm'})
        
        # add trends
        ds = ds.assign(dot_trend=utls.get_fit(ds.dot, 'time', 1))
        ds = ds.assign(dota_trend=utls.get_fit(ds.dota, 'time', 1))
        
        # store
        self.ds = ds
        
    def __load(self,ver): 
        # load data into xarrays
        if ver == "goco":
            data = xr.open_dataset(FNAME_GOCO)
        elif ver =="egm":
            data = xr.open_dataset(FNAME_EGM)
        elif ver == "eig" or ver == "eigen":
            data = xr.open_dataset(FNAME_EIG)
        else:
            print("Data must be one of 'grace', 'egm' or 'eigen'")
        return data
    
    def plot_dota_section(self,time,lat):
        #plot at one lat as well to check anomal is being calculated correctly
        fig, ax = plt.subplots(2,1,figsize=(16,6))
        
        self.ds.dot.interp(time=time).interp(latitude=lat).plot(ax=ax[0],label='DOT at {}'.format(time))
        self.ds.ref.interp(latitude=lat).plot(ax=ax[0],label='MDT (Jan-04 to Dec-09)')
        self.ds.dota.interp(time=time).interp(latitude=lat).plot(ax=ax[1],label='DOTA at {}'.format(time))

        ax[0].legend()
        ax[0].grid()
        ax[1].legend()
        ax[1].grid()
        return fig
        
class SLA():
    def __init__(self):
        
        ds = xr.open_dataset(FNAME_SLA)
        
        # assign coordinates
        ds=ds.assign_coords({'latitude':ds.latitude,'longitude':ds.longitude})
        
        # remove errors
        ds = ds.where(ds.sla<100,drop=True).where(ds.sla>-100,drop=True)
        
        # make monthly
        ds=ds.resample(time='1MS').mean()
        
        # convert to cm
        ds['sla'] = ds.sla * 100
        
        # add trend
        ds = ds.assign(trend=utls.get_fit(ds.sla, 'time', 1))
        
        # store
        self.ds = ds
         
class GRACE():
    def __init__(self,from_file=True,source='CSR',store_raw=False):
        
        if source=='CSR':
            fname = FNAME_DATA_CSR
            fname_mask = FNAME_MASK_CSR
            fname_saved = PROCESSED_FOLDER + 'GRACE_CSR.nc'
        elif source=='JPL':
            fname = FNAME_DATA_JPL
            fname_mask = FNAME_MASK_JPL
            fname_saved = 'GRACE_JPL.nc'
        elif source=='GAD':
            fname= FNAME_DATA_GAD
            fname_mask = FNAME_MASK_CSR
            fname_saved = 'GRACE_GAD.nc'
            
        if from_file:
            self.ds = xr.open_dataset(fname_saved)
        else:
     
            # load data into xarrays
            ds = xr.open_dataset(fname)
            ds_landmask = xr.open_dataset(fname_mask)

            # store
            if store_raw:
                self.ds_raw = ds
                self.landmask = ds_landmask

            # apply land mask
            ds = ds.where(1-ds_landmask.land_mask, drop=True) if source=='JPL' else ds.where(ds_landmask.LO_val, drop=True)

            # smooth before trimming
            #ds['lwe_thickness_smoothed'] = ds.lwe_thickness.rolling(lat=19,lon=46).mean()

            if store_raw:
            # store global ds
                self.ds_global = ds

            # trim to area of analysis
            # do time later as we need to decode julian time
            ds = ds.sel(lat=slice(LAT_MIN,LAT_MAX))

            # all lat/lon vaues are 0-360
            ds = utls.lon_360_to_180(ds,'lon').rename({'lat': 'latitude'})

            # we also want to change the time units to datetimes so we can work more easily with them in xarray
            # (ONLY IF USING CSR)
            ds['time'] = ds.time.assign_attrs({"units": "days since 2002-01-01"})
            ds = xr.decode_cf(ds)
        
            # store whole timeseries
            self.ds_all = ds

            # we want to compare against monthly averages, and since grace is technically daily, resample
            # as the others are monthly averages, get the same for grace
            # MS means it is resampled to the first day of each month. so is exactly in line with other 2
            ds = ds.resample(time='1MS').mean()

            # before trimming time, interpolate in time
            ds['lwe_thickness'] = ds.lwe_thickness.interpolate_na('time',limit=2)

            self.ds = ds

    def plot_times(self):
        # plot all eha over time to get a better idea of how gappy data is
        # seems it would be a bad idea to interp esp. over 2017 since variation is so large. uncertainty would be big
        mds = self.ds.mean(dim='latitude').mean(dim='longitude')
        fig, ax = plt.subplots(figsize=(16,4))
        ax.plot(mds.time,mds.lwe_thickness,'x',label='unsmoothed')
        ax.plot(mds.time,mds.lwe_thickness_smoothed,'x',label='smoothed')
        ax.legend()
        ax.grid()
        
    def plot_mean(self, gbl=False):
        # plot all eha over time to get a better idea of how gappy data is
        ds = self.ds_all if gbl else self.ds
        mds = ds.mean(['latitude','longitude'])
        fig, ax = plt.subplots(figsize=(16,4))
        ax.plot(mds.time,mds.lwe_thickness,label='unsmoothed')
        ax.plot(mds.time,mds.lwe_thickness_smoothed,label='smoothed')
        ax.legend()
        ax.grid()
        
class ERA5():
    def __init__(self):#,start,end):
        # load in data
        ds = xr.open_dataset(FNAME_ERA_GLOBAL)
        
        # rename msl to mslp as this is confusing
        ds = ds.rename({'msl':'mslp'})
        
        # flip along latitude so it increasing
        ds = ds.sortby('latitude')
        
        # store whole timeseries
        self.ds_all = ds
        
        # crop to time of analysis
        #ds = ds.sel(time=slice(start,end))
        
        # average over the global ocean
        mslp = ds.mslp.mean(('latitude','longitude'))
        ds['mslp'] = mslp.assign_attrs({'description':'global average pressure','units':'Pa'})
        
        msl = 100 * ds.mslp * (1 / (RHO * G))
        ds['msl'] = msl.assign_attrs({'description': 'global sea level due to atmospheric pressure','units':'cm'})
        
        # reference
        ref = ds.mslp.mean()
        ds['reference']  = ref.assign_attrs({'description': 'mean global sea level pressure','units':'Pa'})
        
        # anomalies
        mslpa = ds.mslp - ds.reference
        ds['mslpa'] = mslpa.assign_attrs({'description': 'mean global sea level pressure anomaly wrt time mean','units':'Pa'})

        # convert to sea level equivalent and turn into cm
        msla = 100 * ds.mslpa * (1 / (RHO * G))
        ds['msla'] = msla.assign_attrs({'description': 'mean global sea level anomaly due to atmospheric pressure','units':'cm'})
        
        # change longitude coords to -180 to 180
        ds = utls.lon_360_to_180(ds,'longitude')
        
        # add trend
        ds = ds.assign(msla_trend=utls.get_fit(ds.msla, 'time', 1))
               
        # store ds
        self.ds = ds

class GPHA():
    def __init__(self):
        self.profile_df = pd.read_csv(FNAME_PROFILES_DF,index_col=0,parse_dates=True)
        self.gridded_ds = xr.open_dataset(FNAME_PROFILES_GRIDDED)

if __name__ == '__main__':
    dot = DOT('egm')
    t = dot.ds.time.to_numpy()
    print("Retrieved DOT data from {0} to {1}".format(t[0],t[-1]))
