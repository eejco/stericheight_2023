import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import warnings
#import mpl_scatter_density
import cmocean

from utils import *
#from region import Region

# from astropy.visualization import LogStretch
# from astropy.visualization.mpl_normalize import ImageNormalize
# from sklearn.metrics import mean_squared_error
# from shapely import geometry

class PlottingFns():
    def __init__(self):

        self.SEAS = {
            'Southern Ocean': [-180,180,-90,-50],
            'Small Southern Ocean': [-180,180,-90,-60],
            'Big Ross Sea': [140,-150,-80,-65],
            'Ross Sea': [160,-150,-80,-65],
            'Eastern Ross Sea': [160,180,-80,-65],
            'Terranova Bay': [160,170,-78,-74],
            'Ronne Polynya': [-60,-30,-80,-70],
            'ws convective region': [-31,-9,-70,-60],
            'Weddell Sea': [-60,10,-80,-50],
            'Weddell Sea DWF': [-50,-30,-80,-65],
            'Amundsen Sea': [-140,-100,-70,-60],
            'Bellinghausen Sea': [-100,-60,-70,-60],
            'Australian-Antarctic Basin': [70,170,-70,-50],
            'Adelie Coastline': [105, 160, -69, -60],
            'Adelie & George V': [140, 149, -75, -60],
            'Amery': [65,82,-74,-66],
            'Cooperation Sea': [60,90,-80,-60],
            'Maud Rise': [-10,10,-67,-60],
            'Seal Island': [65,80,-55,-50],
            'North of 65S': [-180,180,-65,-50],
            'South of 60S': [-180,180,-90,-60],
            'Larsen B': [-62,-58,-66.1,-65.1],
            'MR View': [-20,25,-75,-58],
            'MRP': [0,10,-67,-62]
            }

    def sp(self,ax,da,vmax=None,vmin=None,title="",cbar=None,cbar_orientation='horizontal',cmap='Spectral_r',sea='Southern Ocean',bathy=None,extent=None,wrapped=False,land_zorder=100):
        """ Create a plot of Antarctica from -50N
        
        reqired inputs
        ax: subplot axes with 'projection': ccrs.SouthPolarStereo()
        da: xr data array
        title: title for plot
        vmax: colorbar max, and -min if no vmin provided
        
        optional inputs
        vmin: optional color minimum if not centred at 0
        cbar: provide cbar label if cbar required, otherwise ignore
        cbar_orientation: 'horizontal' or 'vertical'
        cmap: specify colormap
        sea: 'weddell' or 'ross' for zoomed into weddell and ross sea, otherwise all of antarctica
        bathy: bathymetry data array
        
        returns
        img: handle to image plotted onto axes
        
        """
                
        # ignore warnings
        warnings.filterwarnings('ignore')
        
        if extent is None:
            extent = self.SEAS[sea]

        if (extent[0] > extent[1]) and (extent[1] < 0):
            extent[1] = extent[1] + 360

        if (vmin == None) and (vmax != None):
            vmin = -vmax
         

        # limit ylim and xlim to specified coordinates
        ax.set_extent(extent, ccrs.PlateCarree())
        
        def plot_(da_,vmax):
            if (vmax != None):
                img = da_.plot(ax=ax,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,x='longitude',y='latitude',add_colorbar=False)
            else:
                img = da_.plot(ax=ax,transform=ccrs.PlateCarree(),x='longitude',y='latitude',add_colorbar=False)
            return img

        # assign plot to variable
        if wrapped:
            da_pos = da.where(da.longitude > 0,drop=True)
            da_neg = da.where(da.longitude < 0,drop=True)
            im1 = plot_(da_pos,vmax)
            im1.set_cmap(cmap)
            vmin,vmax = im1.get_clim()
            img = plot_(da_neg,vmax)
        else:
            img = plot_(da,vmax)

        img.set_cmap(cmap)
        
        # add bathymetry if provided
        if bathy is not None:
            #img_bathy = ax.contour(bathy.longitude.values,bathy.latitude.values,bathy.values,transform=ccrs.PlateCarree(),levels=[-4000,-1000],cmap="copper_r",vmin=-10000,vmax=0,linewidths=1.5,linestyles='--')
            img_bathy = ax.contour(bathy.longitude.values,bathy.latitude.values,bathy.values,transform=ccrs.PlateCarree(),levels=[-1000],cmap="copper_r",vmin=-10000,vmax=0,linewidths=1.5,linestyles='--')


        # add colorbar below
        if cbar is not None:
            cbar_obj = plt.colorbar(img,ax=ax,orientation=cbar_orientation)
            cbar_obj.set_label(cbar)

        # add map stuff
        if wrapped:
            # because ross is centred around 180, we need to shift gridlines
            ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=[120,140,160,180,200,220,240])
            ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=[120,140,160,180,-160,-140,-120])
        else:
            ax.gridlines(draw_labels=True)
        ax.add_feature(cartopy.feature.LAND,zorder=land_zorder, edgecolor='k')
        ax.set_title(title)
        
        if (extent[0]==-180) and (extent[1]==180):
            # Compute a circle in axes coordinates, which we can use as a boundary
            # for the map. We can pan/zoom as much as we like - the boundary will be
            # permanently circular.
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)

            ax.set_boundary(circle, transform=ax.transAxes)

        return img
       
    def climatology(self,ax,data,labels,color,marker=[],counts=[],twinx=False,twinx_data=[],twinx_labels=[],twinx_color=[],twinx_marker=[],normalize=False,ylbl='Monthly Mean'):
        """ Plot climatology
        data: list of 1 x 12 da or numpy array corresponding to each month
        labels: corresponding labels to each item in data list
        colors: colour crresponding to each item in data list
        counts: count of each item in data list"""
        def plot_climatology(ts,c,ax,lbl,mrkr,xlims=None):
            if normalize:
                ts = ts - np.nanmean(ts)
            mid = (ts[0]+ts[11])/2

            ln = ax.plot(np.arange(1,13),ts,mrkr,c=c,lw=2,label=lbl)

            ax.plot([12,12.5],[ts[11],mid],':',c=c,lw=2)
            ax.plot([0.5,1],[mid,ts[0]],':',c=c,lw=2)
            return ln
        
        if len(marker)==0:
            marker = ['x-' for x in data]

        xlbl = ['J','F','M','A','M','J','J','A','S','O','N','D']
        
        lns = [plot_climatology(d,c,ax,l,m) for d,c,l,m in zip(data,color,labels,marker)]
        lns2 = []
        
        if twinx:
            ax2=ax.twinx()
            labels = labels + twinx_labels
            if len(twinx_marker)==0:
                twinx_marker = ['x-' for x in twinx_data]
            lns2 = [plot_climatology(d,c,ax2,l,m) for d,c,l,m in zip(twinx_data,twinx_color,twinx_labels,twinx_marker)]
            lns = lns + lns2
            
        if len(counts) > 0:
            if not twinx:
                ax2 = ax.twinx()
                ax2.bar(np.arange(1,13),counts,color='grey',alpha=0.3)
                ax2.set_ylabel('No. profiles')

        yy = ax.get_ylim()
        sepcol = 'darkred'

        add_div = lambda x: ax.plot([x,x],yy,'--',color=sepcol)

        add_div(3.5)
        add_div(6.5)
        add_div(9.5)

        ax.set_xticks(np.arange(1,13))
        ax.set_xticklabels(xlbl)
        ax.set_ylim(yy)
        ax.set_xlim([0.5,12.5])

        if normalize:
            ylbl = 'Normalised ' + ylbl
            
        ax.set_ylabel(ylbl)

        ax.grid()
        ttl = ax.set_title('Summer\tAutumn\tWinter\tSpring'.replace('\t','                                 '))
        
        def flatten(l):
            return [item for sublist in l for item in sublist]
        
        ax.legend(flatten(lns),labels)
        return lns, lns2
    