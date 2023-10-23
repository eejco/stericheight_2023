import xarray as xr
import pandas as pd
import numpy as np

from gsw import density
from tqdm.notebook import tqdm_notebook
from enum import Enum

class Color(Enum):
    '''
    Colour class to define consistent colouring
    LABEL = [dark colour,light colour]
    '''
    MEOP = ['orchid',None]
    ARGO = ['mediumpurple',None]
    GPHA = ['rebeccapurple','thistle']
    SHA = ['teal','paleturquoise']
    DOTA = ['steelblue', 'lightblue']
    BHA = ['chocolate','peachpuff']


class Utils():
    
    def detrend_dim(da, dim, deg=1):
        """ detrend along a single dimension """
        p = da.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(da[dim], p.polyfit_coefficients)
        return da - fit
    
    def get_fit(da, dim, deg):
        """ get fit in a single dimension """
        p = da.polyfit(dim=dim, deg=deg)
        return xr.polyval(da[dim], p.polyfit_coefficients).where(~np.isnan(da))

    def xr2dt(xr_time):
        return pd.to_datetime(xr_time.item())
    
    def lon_360_to_180(ds,lon_label_old='longitude',lon_label_new='longitude'):
        """ change longitude values from 0 to 360 to -180 to 180 """
        # since we are centred around the south pole (0,0) it makes sense to work with coordinates which are -180 to 180
        # first add a new co-ord for new lons
        ds['_longitude_adjusted'] = xr.where(
            ds[lon_label_old] > 180,
            ds[lon_label_old] - 360,
            ds[lon_label_old])
        # reassign the new coords to as the main lon coords
        # and sort DataArray using new coordinate values
        ds = (
            ds
            .swap_dims({lon_label_old: '_longitude_adjusted'})
            .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
            .drop(lon_label_old))

        return ds.rename({'_longitude_adjusted': lon_label_new})
    
    def get_gpha(folder,f,temp='TEMP',psal='PSAL',pres='PRES',as_xr=False):        
        try:
            ds = xr.open_dataset(folder+f)
        except:
            print("Error opening file: {}".format(f))
            return
        
        # calculate rho for each profile/level
        rho = density.rho(ds[psal],ds[temp],ds[pres])
        
        # calculate specific volume for each profile/level
        vs = (1/rho - 1/density.rho(35,0,500))/9.82#ds[pres]))/9.82
        vs = vs.assign_coords({'PRES_Pa': ds[pres]*10000})
        
        # integrate vs to get gpha for each profile
        # TODO deal with NaNs
        gpha = vs.groupby('N_PROF').map(lambda x: x.integrate('PRES_Pa') * 100)
 
        # combine into new dataset omitting depth levels
        if as_xr:
            return xr.DataArray(
                coords={
                    'time': ds.JULD,
                    'longitude': ds.LONGITUDE,
                    'latitude': ds.LATITUDE
                },
                data=gpha
            )
                
        else:
            return pd.DataFrame({
                    'file_id':  f,
                    'float_id' : ds.PLATFORM_NUMBER,
                    'lat' : ds.LATITUDE,
                    'lon' : ds.LONGITUDE,
                    'time' : ds.JULD,
                    'gpha': gpha
                },
            )
        
    def add_elevation(df,elevation):
        get_elev = lambda itm: elevation.interp(latitude=itm['lat'],longitude=itm['lon']).item()
        df['elevation'] =  list(map(get_elev,tqdm_notebook(df.to_dict('record'))))#[get_elev(itm) for itm in df.to_dict('record')]

        return df



