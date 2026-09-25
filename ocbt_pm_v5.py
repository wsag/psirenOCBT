#!/usr/bin/env python3
print("Imports ...")
import sys
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm
#from threading import Thread

print("Loading data ...",end="")
refresh = False

### 2025-08-27 - Updating with new statistics for distance probabilities.
# Can keep omega_v1.3 calculations.
### 2026-07-01 - Include estimates of uncertainty on properties.


def get_ds(year):
    if refresh == True:
        chnl_slope = xr.open_dataset("/net/nfs/merrimack/raid/data/NHDPlusHR_v2/NHDPlus_HR_slope_01min.nc")
        chnl_slope = chnl_slope.rename({"Band1":"slope"})["slope"]*0.01
        #  WE NEED A WAY BETTER WAY OF INFILLING MISSING SLOPE DATA!!!!
        #    can we regress channel slope on topographic slope?
        chnl_slope = xr.where(chnl_slope < 1e-5, 1e-5,chnl_slope)
        chnl_slope = chnl_slope.fillna(1e-5)
        print(chnl_slope)
        print("slope (mmm)",chnl_slope.min(),chnl_slope.mean(),chnl_slope.max(),flush=True)

        forest = xr.open_dataset("IGBP_CONUS_01min_AnyForest_upAvg.asc",engine="rasterio").rename({"band_data":"forest","x":"lon","y":"lat"}).squeeze(dim="band") # in fraction
        urban  = xr.open_dataset("GISA_CONUS_01min_UrbImperv_upAvg.asc", engine="rasterio").rename({"band_data":"urban","x":"lon","y":"lat"}).squeeze(dim='band') # in fraction 

        #  NOTE: contents of the wbm_dir include 21 yearly files each at 45 Gb.  10 years are used in this analysis
        #        Please contact authors if you would like to recreate this analysis to arrange receipt of these data,
        #        otherwise, substitute continuous, time-varying fields of discharge and stream width from your preferred
        #        data source.
        wbm_dir = "/net/nfs/swift/raid2/data/psiren/ocbt/wbm/output/ocbt_conus_v1/daily/"
        
        # Assume constant sinuosity of 1.2
        length = 1200. * xr.open_dataset("/net/nfs/swift/raid2/data/MERIT/MERIT_Hydro_IHU/01min/CONUS/MERIT_plus_60sec_CONUS_lkmsk_clean_mouthDist.asc",engine="rasterio")["band_data"] # 
        length = length.squeeze('band')
        length = length.rename({"x":"lon","y":"lat"})
    
        print("length (mmm)",length.min(),length.mean(),length.max())
        print(length)

        ds = xr.open_mfdataset([wbm_dir+"wbm_{}.nc".format(year)],data_vars="discharge width".split())

        ds["slope"] = chnl_slope.interp_like(ds)
        ds["forest"] = forest["forest"].interp_like(ds)
        ds["urban"]  = urban["urban"].interp_like(ds)
        ds["length"] = length.interp_like(ds).rename("length")
        print(ds)
        ds["omega"] = xr.where(ds["width"] > 0.,9999. * 9.81 * ds["discharge"].values * ds["slope"].values / ds["width"].values,0.).compute()
        na={"_FillValue":-9999.}
        encoding = dict([(k,na) for k in ds.data_vars])
        mask = ds["discharge"] == ds["discharge"]
        print(mask)
        ods = xr.where(mask,ds,np.nan)
        ods["crs"] = ds["crs"]
        print(ods)
        ds.to_netcdf("./local/ocbt_conus_v1.3_omega0_{}.nc".format(year),encoding=encoding) # /forcings/ if want fast write to rapid
    if refresh == "RefreshLength":
        ds = xr.open_dataset("./local/ocbt_conus_v1.2_omega0_{}.nc".format(year))
        length = 1200. * xr.open_dataset("/net/nfs/swift/raid2/data/MERIT/MERIT_Hydro_IHU/01min/CONUS/MERIT_plus_60sec_CONUS_lkmsk_clean_mouthDist.asc",engine="rasterio")["band_data"] # 
        length = length.squeeze('band')
        length = length.rename({"x":"lon","y":"lat"})
        ds["length"] = length.interp_like(ds).rename("length")
        na={"_FillValue":-9999.}
        encoding = dict([(k,na) for k in ds.data_vars])
        ds.to_netcdf("./local/ocbt_conus_v1.3_omega0_{}.nc".format(year),encoding=encoding) # /forcings/ if want fast write to rapid
        
    else:

       ds = xr.open_dataset("./local/ocbt_conus_v1.3_omega0_{}.nc".format(year))

    ds = ds.loc[{"time":slice("{}-01-01".format(year),"{}-12-31".format(year))}]

    print("... data loaded")
    return ds    

def calculate_daily_probability(ds,year,lodged_d,P_memory,estimate="center"):
    assert estimate in "center lower upper".split()
    print(ds.coords)
   
    print("Clipping inputs to target ranges ...",end="",flush=True)
    ds["forest"] = ds["forest"].clip(min=1e-5)
    ds["urban"]  = ds["urban"].clip(min=1e-5)
    ds["discharge"] = ds["discharge"].clip(min=1e-7)
    ds["omega"] = ds["omega"].clip(min=1e-7)
    print("... done",flush=True)
    """ We are looking for the probability that a piece of floating trash is mobilized out of the pixel each day.
    This is given by the probability that the bottle travels the length of streams in the pixel conditioned on 
    the probability that it is mobilized at all.
    D: Distance mobilized (m)
    L: Length of streams in pixel (m)
    M: Mobilization state (0=stationary, 1=mobilized)
    This is given by P(D>L) = P(D>L|M=1)P(M=1) + P(D>L|M=0)P(X=0)
    We assume that the probability that a bottle leaves the pixel when the bottle isn't mobilized [P(D>L|M=0)] = 0, 
    since we never observed this.  Therefore, we have only: P(D>L) = P(D>L|M=1)P(M=1).
    #    D: distance mobilized each day is given by exp(4.052±0.058 + 0.325±0.023 ln(Q) - 0.0902±0.011 ln(f_forest%)) # superceded
    # updated 2025-08-27
    D: distance mobilized each day: exp(7.571±.31+0.4356(±.031)(ln(Q))-7.5994(±1.121)(sqrt(f_for))+2.472(±.404)(ln(f_urb))-5.5624(±1.194)(sqrt(f_for)xln(f_urb))

    Q: Discharge m3s
    f_forest%: Upstream forest cover (%)
    f_urban%: Upstream urban cover (%)
    L: Sn A_p**0.5
    Sn: Sinuosity = 1.2
    A_p: Area of pixel in m2.
    
    Define B = ln(D)-ln(L) so that B>0 when D>L (when distance mobilized is Beyond the lengh of the reach)
    Then the expected value for B each day is:
    E(B) = 7.5709-ln(L) + 0.4356 ln(Q) - 7.5994 (f_for)**0.5 + 2.4719 ln(f_urb) - 5.5624 (f_for**0.5)xln(f_urb) ,  and variance in B is
    V(B) = MSE(1 + 1/n+ x_i.T (X.T X)^-1 x_i) Where (X.T X)^-1 is the inverse design matrix, x_i is the new vector of 
           of independent estimators, where MSE is the mean squared error of the residuals regression (3.138 ln(m2)) and n is the number of observations used
     in the regression (840).
    Then we have P(B>0) = P( (B-E(B)/V(B)^0.5) > (0-E(B)/V(B)^0.5) )  or
    P(B>0) = P(D>L|M=1,Q,f_for,f_urb) = 1 - \Phi (0-E(B)/V(B)^2) where \Phi is the cumulative standard normal.
    
    Then we evaluate the probability that an item is actually mobilized on a given day.
    P(M=1|omega0) = 1/1+exp(-(-3.91±0.031+0.083±0.01*ln(omega0)))
    
    Now this gives me the probability I seek directly, but it doesn't account for error in the estimation so the update 
     provides the ability to account for upper and lower estimates of P(M) accounting for regression error.

    These define my daily probabilities of trash leaving the pixel.  
    """
    shape = ds["discharge"].shape
    print(shape)

    s2 = 3.1379432 # residual variance (ln(m2))
    n = 840

    # from ocBottle_Probability_Model_distance.ipynb where X is the design vector
    X_inv = np.array( [[ 2.64387434e-02,  4.21417648e-05, -8.61754469e-02,  2.77251104e-02, -8.49824165e-02],
                       [ 4.21417648e-05,  2.69074523e-04, -6.30462090e-05,  3.25485480e-04, -1.02238057e-03],
                       [-8.61754469e-02, -6.30462090e-05,  3.46591192e-01, -8.23754760e-02,  3.18110608e-01],
                       [ 2.77251104e-02,  3.25485480e-04, -8.23754760e-02,  4.49777199e-02, -1.19481095e-01],
                       [-8.49824165e-02, -1.02238057e-03,  3.18110608e-01, -1.19481095e-01,  3.92889148e-01]])

    #  So the variance calculation can't handle the full year and domain.  Break it down by day:
    P_B0 = []
    print("   Calculating probability of distance greater than pixel",flush=True)
    print(" Day: ",end="",flush=True)
    for d in range(shape[0]):
        print ("{} ..".format(d),end="",flush=True)
        E_B= 7.5709 + 0.4356 * np.log(ds["discharge"].values[d,:,:]) - 7.5994 * (ds["forest"].values[:,:]**0.5) + 2.4719 * np.log(ds["urban"].values[:,:]) - 5.5624 * ((ds["forest"].values **0.5) * np.log(ds["urban"].values[:,:])) - np.log(ds["length"].values)
        # There is a more elegant way of doing this with the X_inv (but with our new L term).
        
        x = np.stack( ( np.ones_like(ds["discharge"].values[d,:,:].flatten()), np.log(ds["discharge"].values[d,:,:].flatten()),
                        ds["forest"].values.flatten()**0.5, np.log(ds["urban"].values.flatten()),   # These need to be stacked over the 0th dimension
                        (ds["forest"].values.flatten()**0.5)*np.log(ds["urban"].values.flatten()) ) ).T
        x_mean = np.array([1.0, -0.19683044, 0.3168089,-0.74391234,-0.26695202])
    
        mod_s2 = np.einsum('ij,jk,ik->i',(x-x_mean),X_inv,(x-x_mean))
        V_B = s2 * (1 + 1/n + mod_s2)
        P_B0.append(1 - norm.cdf((0-E_B.flatten())/np.sqrt(V_B)))
    print("")
    print("  Contenating year of results.",flush=True)
    P_B0 = np.concatenate(P_B0).flatten()
    
    print(P_B0.shape)
    P_B0 = P_B0.reshape(shape)
    print(P_B0.shape)

    # !!!!!!   Need to gracefully read in P_memory and lodged_d from disk for beginning of year  !!!!
    #   If these don't exist - then they need to be initialized!!!
    """
    Update 2026-07-07 (SZ): Updating to incorporate uncertainty in Logistic Regression model.  
     We account for the influence of 95% confidence interval on predictor uncertainty at local values of  \omega and \ds.
     We repeat the model run for upper and lower bounds (center is already performed).
    For each day we calculate the error term and modify our estimate of P(M)

    The variance term V is given for each day as:
    V = x^T \Sigma_\beta x
    where x is the feature vector, and
     \Sigma_\beta is the covariance matrix of the regression estimate.
    
    """

    SigB = np.array( [[1.05086101e-04, 2.80437507e-07, 9.96094049e-05],
                      [2.80437507e-07, 2.85292149e-06, 9.13180423e-06],
                      [9.96094049e-05, 9.13180423e-06, 1.07385297e-03]] )
    P_M = np.ones(ds['omega'].values.shape)
    print("Calculating probability of mobilization ...")
    print("Day: ")
    for i,(t,data) in enumerate(ds.groupby('time',squeeze=False)):
        print("{} ..".format(i),end=" ")
        data = data.squeeze()
        dshape=data["omega"].values.shape
    
        V=np.zeros_like(lodged_d.flatten())
        if estimate == "lower":
            coeff = -1.96
        elif estimate == "upper":
            coeff = 1.96
        elif estimate == "center":
            coeff = 0.0

        if estimate != "center":
            x = np.stack( ( np.log( data["omega"].values.flatten()), np.log(lodged_d.flatten()), np.ones_like(lodged_d.flatten()) ) ).squeeze().T
            V = np.einsum('ij,jk,ik->i',x,SigB,x)
                        
        err = (coeff*np.sqrt(V)).reshape(dshape)
        P_today = 1./(1. + np.exp(-(-3.725 + 0.0751 * np.log(data["omega"].values) - 0.0723 * np.log(lodged_d))+err))
        P_M[i] = P_today
        P_memory = P_memory*(1.-P_today)
        U = np.random.rand(1)
        Moved = (P_memory > U).nonzero()
        Unmoved = (P_memory <= U).nonzero()
        P_memory[Moved] = 1
        lodged_d[Moved] = 1
        lodged_d[Unmoved] += 1
        
    
    #### !!!! Need to save out lodged_d and P_memory for next year!!!

    print("      ... and final probability of moving to coast each day",flush=True)
    P_DL = P_B0 * P_M
    print("P_DL shape ",P_DL.shape)
    print(ds.coords)
    print("      ... done",flush=True)
    P_DL = xr.DataArray(data=P_DL,coords=ds.coords,name="Prob_DL")
    P_DL = P_DL.where(ds["discharge"]==ds["discharge"])
    encoding={'Prob_DL':{'_FillValue':-9999}}

    return P_DL, lodged_d, P_memory
    
    
def probability_over_time(p_daily,year,estimate, resampling_epoch):
    """
    We assume that the probability of moving beyond the pixel each day is independent of prior days, 
     such that we accumulate the product of the complements (probability that the trash stays
     in the pixel each day) over the resampling period:
    P(B_resample) = 1-\Pi_0^n (1-P_daily)

    This is a simplification.  There are probabilities that bottles move a distance less than reach length throughout the month
     resampling period.  Therefore, the probability of traveling beyond the reach later in the sampling period increases.
    
    I could recast the daily function to assess probabilities of moving beyond different fractions of the total reach length. 
     Then I could formulate some smart way to increment the probabilities as throughout the resampling period. 


    """
    print("Accumulating probabilility over resampling interval (\' {} \') ...".format(resampling_epoch),end="")
    
    cp_daily = 1.0 - p_daily

    print(np.isnan(cp_daily).sum())
    if resampling_epoch == "woy":
        CP_resample = cp_daily.groupby({"time":[[x.year for x in cp_daily["time"]],[x.weekofyear for x in cp_daily["time"]]]}).prod(skipna=False)
    else:
        CP_resample = cp_daily.resample({"time":resampling_epoch}).prod(skipna=False)
    print(np.isnan(CP_resample).sum())
    P_resample = 1.0 - CP_resample
    print(np.isnan(P_resample).sum())
    encoding={'Prob_DL':{'_FillValue':-9999}}
    print(" ... done.  Saving file ...",end="")
    P_resample.to_netcdf("./P_mobilized_v5_{}_{}_{}.nc".format(resampling_epoch, estimate,year),encoding=encoding)
    print(" ... done.")

def main(res,est):
    years = range(2005,2015) 

    year = years[0]
    print("starting year {}".format(year),flush=True)
    ds = get_ds(year)
    print(ds.coords)
    lodged_d = np.ones(ds['omega'].shape[1:],dtype=np.uint8)
    P_memory = np.ones(ds['omega'].shape[1:],dtype=float)
    probability_mobilized_each_day,lodged_d,P_memory = calculate_daily_probability(ds,year,lodged_d,P_memory,estimate=est)
    probability_over_time(probability_mobilized_each_day,year,est,res)
    for year in years[1:]:
        ds = get_ds(year)
        probability_mobilized_each_day,lodged_d,P_memory = calculate_daily_probability(ds,year,lodged_d,P_memory)
        probability_over_time(probability_mobilized_each_day,year,est,res)

if __name__ == "__main__":
    args=sys.argv[1:]
    
    if len(args) == 2:
        res=args[0]
        est=args[1]
    if len(args) == 0:
        res="ME"
        est="center"
    if len(args) == 1:
        if args[0] in "center lower upper".split():
            est = args[0]
            res = "ME"
        else:
            res = args[0]
            est = "center"
    main(res,est)
        
