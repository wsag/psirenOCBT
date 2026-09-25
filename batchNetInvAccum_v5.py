#!/bin/env python3

import sys
import os
import xarray as xr
import rioxarray
import subprocess
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool

P_base = lambda y,l,r: "P_mobilized_v5_{}_{}_{}.nc".format(r,l,y)
years = range(2006,2015)
resampling_offsets = ["D","ME","QE","YE"][::-1] #.split() # "D YE".split() # "ME YE W".split()
resampling_offsets = ["YE"]
levels="upper lower".split()
endomask = xr.open_dataset("/net/nfs/swift/raid2/data/MERIT/MERIT_Hydro_IHU/01min/CONUS/MERIT_plus_60sec_CONUS_lkmsk_clean_IDs_EnR.asc",engine="rasterio").rename({"band_data":"P_X","x":"lon","y":"lat"}).squeeze(dim="band").isel(lat=slice(None,None,-1))

print(endomask.min(),endomask.max())
      
def make_init(date,level,res,band):
    year=date[:4]
    with open("./networkTools_upsInvAccMin_v5_template.init",'r') as fin:
        lines=fin.readlines()

    rep_lines = [line.replace("<DATE>",date).replace("<YEAR>",str(year)).replace("<BAND>",str(band)).replace("<RES>",res).replace("<LEVEL>",level) for line in lines]
    init_name="./invaccmin_init/networkTools_upsIAP_{}_{}_{}.init".format(res,level,date)
    with open(init_name,'w') as fout:
        fout.writelines(rep_lines)
    return init_name

def proc_year(rly):
    res, level, year = rly
    print(P_base(year,level,res))
    ds = xr.open_mfdataset(P_base(year,level,res))
    for b,dt in enumerate(ds["time"].to_index().date):
        in_nm = make_init(dt.isoformat(),level,res,b+1)
        subprocess.run(["./networkTools",in_nm],capture_output=True)
        out_nm="./invaccmin_dt/P_trans2ocean_v5_{}_{}_{}".format(res,level,dt.isoformat())
        #subprocess.run(["gdal_translate","-of","netcdf",out_nm+".asc",out_nm+".nc"])
        out = xr.open_dataset(out_nm+".asc",engine="rasterio").rename({"band_data":"P_X","x":"lon","y":"lat"}).squeeze(dim="band").isel(lat=slice(None,None,-1))
        out = xr.where(endomask["P_X"].values > 0,0.0,out) # * endomask["P_X"].values
        encoding={"P_X":{"_FillValue":-9999}}
        out.to_netcdf(out_nm+".nc",encoding=encoding)
    return year

if __name__ == "__main__":
    with Pool(processes=len(years)) as pool:
        scens = list(product(resampling_offsets,levels,years))
        print(scens)
        years = pool.map(proc_year, scens )
        print(years)
    
    
    
                
    
