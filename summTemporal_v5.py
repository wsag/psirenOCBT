#!/usr/bin/env python3
import os
import datetime
import pandas as pd
import xarray as xr

def main(rs,level=None):
    if level == None:
        vrs = "v4"
    elif level in "upper lower".split():
        vrs = "v5"
        
    result={'YE':{1:([],[])},
            'QE':{1:([],[]),2:([],[]),3:([],[]),4:([],[])},
            'ME':dict([(i,([],[])) for i in range(1,13)]),
            'D':dict([(i,([],[])) for i in range(1,367)])}[rs]

    whichk={'YE':lambda x: 1,
            'QE':lambda x: {"03":1,"06":2,"09":3,"12":4}[x.split('_')[-1].split('.nc')[0].split('-')[1]],
            'ME':lambda x: int(x.split('_')[-1].split('.nc')[0].split('-')[1]),
            'D': lambda x: pd.to_datetime(x.split('_')[-1].split('.nc')[0]).dayofyear}[rs]
    
    dater = lambda x: datetime.datetime.strptime(x.split('_')[-1].split('.nc')[0],"%Y-%m-%d")
    for fn in os.listdir("./invaccmin_dt/"):
        if (vrs == "v5"):
            if (vrs in fn) & (rs in fn) & (level in fn) & fn.endswith('.nc'):
                print(fn)
                result[whichk(fn)][0].append(dater(fn))
                result[whichk(fn)][1].append(xr.open_dataset("./invaccmin_dt/"+fn))
        elif (vrs == "v4"):
            if (vrs in fn) & (rs in fn) & fn.endswith('.nc'):
                print(fn)
                result[whichk(fn)][0].append(dater(fn))
                result[whichk(fn)][1].append(xr.open_dataset("./invaccmin_dt/"+fn))

                
    print("Make the Probability data arrays")

    encoding={"P_X":{"_FillValue":-9999}}
    PXs={}    
    for epoch,(dates,data) in result.items():
        print(epoch)
        PXs[epoch] = xr.concat(data,pd.DatetimeIndex(dates,name="time"))
    output=xr.concat([x.quantile([0.025,0.5,0.975],dim="time") for x in PXs.values()],pd.Index(PXs.keys(),name="epoch")).max(dim="epoch")
    if (vrs == "v4"):
        output.to_netcdf("./P_X_ocbt_pm_{}_{}.nc".format(vrs,rs),encoding=encoding)
    elif (vrs == "v5"):
        output.to_netcdf("./P_X_ocbt_pm_{}_{}_{}.nc".format(vrs,rs,level),encoding=encoding)
    
if __name__ == "__main__":
    import sys
    resampling = sys.argv[1]
    if len(sys.argv) > 2:
        level = sys.argv[2]
    else:
        level = None
    main(resampling,level)
    
