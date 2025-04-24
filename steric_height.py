import xarray as xr

from utils import Utils as utls

FNAME_GEBCO = '../data/GEBCO/gebco_2021_sub_ice_n-50.0_s-90.0_w-180.0_e180.0.nc'

class StericHeight():
    def __init__(self,                 
                 ssh_ref: str, # DOT or SLA
                 ssh: xr.DataArray, # DOT or SLA
                 msl: xr.DataArray, # atmospheric MSL from ERA5
                 lwe: xr.DataArray, # lwe from GRACE
                 use_grace_grid=False
                 ):
        
        lwe, msl = self.crop_time(ssh,lwe,msl)

        self.ssh_ref=ssh_ref
        self.use_grace_grid=use_grace_grid
        
        if ssh_ref == 'DOT':
            self.ds = self.build_from_DOT(lwe,msl,ssh)
        elif ssh_ref == 'SLA':
            self.ds = self.build_from_SLA(lwe,msl,ssh)

    def get_sha(self,startend=[None,None],crop=False):#,grace_only=False):

        ds = self.ds
        
        # if no start/end date provided, make sure we are taking the average over complete years so that certain months are not emphasised
        start_date = startend[0] or ds.time[len(ds.time) % 12]
        end_date = startend[-1] or ds.time[-1]

        ds_fullyear = ds.sel(time=slice(start_date,end_date))
          
        # compute anomalies
        calc_anomaly = lambda var: ds[var] - ds_fullyear[var].mean('time',skipna=True)
        ds['msla'] = calc_anomaly('msl')
        ds['ssha'] = calc_anomaly('ssh')
        ds['eha'] = calc_anomaly('lwe')

        #if not grace_only:
        #sha = ds.ssha - (ds.eha - ds.msla)
        #ds = ds.assign(sha=sha-sha.mean('time'))#lambda x: x.ssha - (x.eha - x.msla))
        #ds.sha.attrs = {"description": "steric height (sh) = ssh - (eha - msla)"}
        #else:
        sha = ds.ssha - ds.eha
        ds=ds.assign(sha=sha - sha.mean('time'))#lambda x: x.ssha - x.eha)
        ds.sha.attrs = {"description": "steric height (sh) = ssh - eha"}

        if crop:
            ds = ds.sel(time=slice(start_date,end_date))
        
        return ds

    def build_from_DOT(self,lwe: xr.DataArray,msl: xr.DataArray,dot:xr.DataArray) -> xr.Dataset:
                       
        if self.use_grace_grid:
            
            coords = dict(
                    time = lwe.time,
                    latitude = lwe.latitude,
                    longitude = lwe.longitude
                )
            
            dot = dot.interp(latitude=coords['latitude'].values,longitude=coords['longitude'].values)
            
        else:

            coords = dict(
                time = dot.time,
                latitude = dot.latitude,
                longitude = dot.longitude,
            )

            lwe = lwe.interp(latitude=coords['latitude'].values,longitude=coords['longitude'].values)
            
        # create new ds for our regridded data
        ds = xr.Dataset(
            data_vars=dict(
                ssh = dot,
                lwe = lwe,
                msl = msl
            ),
            coords=coords
        )

        return ds
        
        
    def build_from_SLA(self, lwe, msl, sla) -> xr.Dataset:
        
        print('Interpolating to curvilinear DOT grid')
                                   
        # get coords as np
        x=sla.x.values
        y=sla.y.values

        # contstruct coordinate arrays
        lat = xr.DataArray(sla.latitude.values, dims=["x","y"],coords={"x": x, "y": y})
        lon = xr.DataArray(sla.longitude.values,dims=["x","y"],coords={"x": x, "y": y})

        lwe = lwe.interp(latitude=lat, longitude=lon)
        msl = msl.interp()

        coords = dict(
                time = sla.time,
                latitude = lat,
                longitude = lon
        )

        # create new ds for our regridded data
        ds = xr.Dataset(
            data_vars=dict(
                ssh = sla,
                lwe = lwe,
                msl = msl
            ),
            coords=coords
        )
            
        return ds
        
    def crop_time(self,a,b,c):
        # trim to time of analysis as determined by 'a'
        tstart = a.time[0]
        tend = a.time[-1]
        
        b = b.sel(time=slice(tstart,tend))
        c = c.sel(time=slice(tstart,tend))
        return b, c
        
    def load_gebco(self):
        """ Load GEBCO Under-Ice Bathymetry data, interpolate to sha lat and lon and save into ds """
        gebco = xr.open_dataset(FNAME_GEBCO)
        
        # interpolate to shared lat and lon
        gebco = gebco.interp(lat = self.ds.latitude, lon = self.ds.longitude)
        
        # assign bathy data to ds
        ds = self.ds.merge(gebco)
        ds = ds.reset_coords(names=['lat','lon'],drop=True)
        
        self.ds = ds
        return self
