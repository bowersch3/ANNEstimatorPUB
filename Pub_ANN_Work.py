#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:10:41 2024

@author: bowersch
"""
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, '/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/')

from trying3 import check_for_mp_bs_WS,convert_to_datetime,convert_to_date_2,plot_mp_and_bs,\
    plot_MESSENGER_trange_cyl,convert_to_date_2_utc,get_mercury_distance_to_sun,\
        convert_datetime_to_string,plot_MESSENGER_trange_3ax,get_aberration_angle,\
            get_day_of_year,convert_to_date_s

#from generate_all_plot import shade_in_ax
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import r2_score
#from create_lobe_time_series import distance_point_to_curve
import datetime
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
import os



# Specify your save path for files

#save_path = '/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/'

save_path = '/Users/bowersch/Desktop/Python_Code/PUB_ANN_Test/'

# Specify the where the Sun2023 files are located

file_mp_in = '/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/MagPause_In_Time_Duration_ver04_public_version.txt'
file_mp_out = '/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/MagPause_Out_Time_Duration_ver04_public_version.txt'
file_bs_in = '/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/Bow_Shock_In_Time_Duration_ver04_public_version.txt'
file_bs_out = '/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/Bow_Shock_Out_Time_Duration_ver04_public_version.txt'

# Define where the MESSENGER data from the PDS is stored on your machine, used in
# load_MESSENGER_into_tplot

file_MESSENGER_data = '/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated - avg'



def shade_in_transitions(transitions,ax,df):
    
    '''Shade in time series with Mercury's plasma environment determined by Sun2023'''
    y=np.array([-1000,1000])
    
    colors=['firebrick','royalblue','gold','orange','mediumpurple']
    ALPH=[.08,.08,.08,.08,.18]
    
    for i in range(len(transitions)-1):
        
        si=transitions[i]
        
        fi=transitions[i+1]
            
        ax.fill_betweenx(y,df.time.iloc[si],df.time.iloc[fi],color=colors[int(df.Type_num.iloc[si])-1],alpha=ALPH[int(df.Type_num.iloc[si])-1])   
def generate_crossing_dataframe(cross,typ):
    
    '''Create dataframe with all boundary crossings from Sun2023 list
        
    '''
    import numpy as np
    import pandas as pd
    from trying3 import convert_to_datetime
    
    cross_start=cross[0,:]
    
    cross_end=cross[1,:]
    
    cs=np.array([convert_to_datetime(d) for d in cross_start])
    
    ce=np.array([convert_to_datetime(b) for b in cross_end])
    
    cross_df=pd.DataFrame(data={'s':cs,'e':ce})
    
    cross_df['Type']=typ
    
    
    return cross_df
def read_in_Weijie_files():
    
    def convert_from_txt_to_date(file):
        x_in=np.loadtxt(file,usecols=(0,1,2,3,4,5))
        x_out=np.loadtxt(file,usecols=(6,7,8,9,10,11))
        date_in=np.array([])
        date_out=np.array([])
        for i in range(np.size(x_in[:,0])):
            
            if int(np.floor(x_in[i,5])) >= 60:
                x_in[i,5]=0.0
                x_in[i,4]=x_in[i,4]+1
                
            if int(np.floor(x_out[i,5])) >= 60:
                x_out[i,5]=0.0
                x_out[i,4]=x_out[i,4]+1
                
            if int(np.floor(x_out[i,5])) < 0:
                x_out[i,5]=59
                x_out[i,4]=x_out[i,4]-1
                
                if x_out[i,4]<0:
                    x_out[i,3]=x_out[i,3]-1
                    x_out[i,4]=59
                
            
            if int(np.floor(x_in[i,5])) < 0:
                x_in[i,5]=59
                x_in[i,4]=x_in[i,4]-1
                if x_in[i,4]<0:
                    x_in[i,3]=x_in[i,3]-1
                    x_in[i,4]=59
            
            date_in=np.append(date_in,str(int(np.floor(x_in[i,0])))+'-'+str(int(np.floor(x_in[i,1])))+'-'+str(int(np.floor(x_in[i,2])))+' '+str(int(np.floor(x_in[i,3])))+
                              ':'+str(int(np.floor(x_in[i,4])))+':'+str(int(np.floor(x_in[i,5]))))
            date_out=np.append(date_out,str(int(np.floor(x_out[i,0])))+'-'+str(int(np.floor(x_out[i,1])))+'-'+str(int(np.floor(x_out[i,2])))+' '+str(int(np.floor(x_out[i,3])))+
                               ':'+str(int(np.floor(x_out[i,4])))+':'+str(int(np.floor(x_out[i,5]))))
                                                                
            
            
        date=np.array([date_in,date_out])
        
        return date
    
    mp_in=convert_from_txt_to_date(file_mp_in)
    mp_out=convert_from_txt_to_date(file_mp_out)
    bs_in=convert_from_txt_to_date(file_bs_in)
    bs_out=convert_from_txt_to_date(file_bs_out)
    
    return mp_in, mp_out, bs_in, bs_out  

def create_full_data_pickle():
    ''' 
        Create a dataset of MAG, ephemeris, and region based on Sun2023 
        boundary crossing list. Returns a pickle with all the data loaded,
        to be used to construct the training/test dataset etc.
        
        Loads each day of MESSENGER data with load_MESSENGER_into_tplot
        
        '''
        
    import pandas as pd
    import numpy as np
    
    import matplotlib.pyplot as plt
    import matplotlib.axes as ax
    
    
    import datetime
    import matplotlib.dates as mdates
    
    from scipy.optimize import curve_fit
    
    from trying3 import convert_to_datetime,convert_datetime_to_string, load_MESSENGER_into_tplot,read_in_Weijie_files, check_for_mp_bs_WS,plot_mp_and_bs
    
    
    R_m=2440
    
    mp_in, mp_out, bs_in, bs_out = read_in_Weijie_files()
    
    month=[31,28,31,30,31,30,31,31,30,31,30,31]
        
        #month=[10]
        #m_number=['05']
        
        
        # Data gaps=
    data_gaps=['2012-06-9','2012-06-10','2012-06-11','2012-06-12', \
                   '2012-06-13','2013-01-8','2013-01-9','2013-02-28', \
                       '2011-04-5','2011-05-24','2011-05-25','2011-05-26',\
                          '2011-05-27','2011-05-28','2011-05-29','2011-05-30',
                          '2011-05-31','2011-06-1','2011-06-2','2011-06-3','2014-12-26']
            
    
    
    years=['2011','2012','2013','2014','2015']
    
    
    ephx_total=np.array([])
    
    ephy_total=np.array([])
    
    ephz_total=np.array([])
    
    magx_total=np.array([])
    
    magy_total=np.array([])
    magz_total=np.array([])
    
    time_total=np.array([])
    
    magamp_total=np.array([])
    
    #Run through each month and day of the year specified
        
    for i in years:
        
        m_number=['01','02','03','04','05','06','07','08','09','10','11','12']
        
        if i=='2015': m_number=['01','02','03','04']
            
        if i=='2011':m_number=['03','04','05','06','07','08','09','10','11','12']
        
        
        
        i_m_number=[str(d) for d in m_number]
    
        for k in i_m_number:
            mk=month[int(k)-1]
            start=1
            if ((i=='2011') & (k=='03')):
                start=24
                
            if ((i=='2015')& (k=='04')):
                mk=29
                
            mn=str(k)
            
        
            for j in range(start,mk+1):
                    #j=j+1
                    
                    
                    
                date_string=i+'-'+mn+'-'+str(j)
                    
                if date_string not in data_gaps:
                    
                    print(date_string)
                    
                    #date_string='2015-01-11'
                    
                    
                    time,mag,magamp,eph=load_MESSENGER_into_tplot(date_string)
                    
                    eph=eph/R_m
                    time=np.array(time)
                    
                    time_total=np.append(time_total,time)
                    
                    ephx_total=np.append(ephx_total,eph[:,0])
                    
                    ephy_total=np.append(ephy_total,eph[:,1])
                    
                    ephz_total=np.append(ephz_total,eph[:,2])
                    
                    
                    magx_total=np.append(magx_total,mag[:,0])
                    
                    magy_total=np.append(magy_total,mag[:,1])
                    
                    magz_total=np.append(magz_total,mag[:,2])

                    
                    magamp_total=np.append(magamp_total,magamp)
                    
                        
    df=pd.DataFrame(data={'time':time_total,'ephx':ephx_total,'ephy':ephy_total,\
                          'ephz':ephz_total,'magx':magx_total,'magy':magy_total,\
                              'magz':magz_total,'magamp':magamp_total})
            
    pd.to_pickle(df,save_path+'full_data_2.pkl')
    
    def assign_training_data():
        #all_data=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/full_data.pkl')
        
        all_data=pd.read_pickle(save_path+'full_data_2.pkl')
        mp_in, mp_out, bs_in, bs_out = read_in_Weijie_files()
        mi=generate_crossing_dataframe(mp_in,'mpi')
        
        mo=generate_crossing_dataframe(mp_out,'mpo')
        
        bi=generate_crossing_dataframe(bs_in,'bsi')
        
        bo=generate_crossing_dataframe(bs_out,'bso')
        
        crossings=[mi,mo,bi,bo]
        
        cc=pd.concat(crossings)
        
        c=cc.sort_values('s')
        
        all_data['Transition']=True
        
        all_data['Type_num']=0
        
        for i in range(np.size(c['s'])-1):
        #for i in range(5980,6000):
            start_time=c['e'].iloc[i]#+datetime.timedelta(minutes=5)
            
            end_time=c['s'].iloc[i+1]#-datetime.timedelta(minutes=5)
            
            gd=np.where((all_data.time > start_time) & (all_data.time < end_time))[0]
            
            type1=c['Type'].iloc[i]
            
            type2=c['Type'].iloc[i+1]
           
            #print(c['e'].iloc[i])
            
            gd_t=np.where((all_data.time >= c['s'].iloc[i])& (all_data.time <= c['e'].iloc[i]))[0]
            
            if np.size(gd)>0:
                if (type1=='bso' and type2=='bsi'):
                    #Starting with an outbound bowshock crossing, and ending with inbound,
                    # so, time in between is in the solar wind:
                        
                    all_data['Type_num'].iloc[gd]=3
                    
                    all_data['Transition'].iloc[gd]=False
                    
                    if np.size(gd_t)> 0:
                        all_data['Type_num'].iloc[gd_t]=4
                    
                    
                if (type1=='bsi' and type2=='mpi'):
                    
                    # Starting with inbound bowshock and ending with inbound magnetopause,
                    # so, time in between is magnetosheath
                    
                    all_data['Type_num'].iloc[gd]=2
                    
                    all_data['Transition'].iloc[gd]=False
                    
                    if np.size(gd_t)> 0:
                        all_data['Type_num'].iloc[gd_t]=4
                    
                if (type1=='mpi' and type2 == 'mpo'):
                    
                    # Starting with inbound magnetopause and ending with outbound magnetopause,
                    # so, time in between is in the magnetosphere
                    
                    all_data['Type_num'].iloc[gd]=1
                    
                    all_data['Transition'].iloc[gd]=False
                    
                    if np.size(gd_t)> 0:
                        all_data['Type_num'].iloc[gd_t]=5
                    
                if (type1=='mpo' and type2 == 'bso'):
                    
                    # Starting with outbound magnetopause and ending with outbound bowshock,
                    # so, time in between is magnetosheath
                    
                    all_data['Type_num'].iloc[gd]=2
                    
                    all_data['Transition'].iloc[gd]=False
                    
                    if np.size(gd_t)> 0:
                        all_data['Type_num'].iloc[gd_t]=5
                
                print(c['e'].iloc[i])
                
        pd.to_pickle(all_data,save_path+'fd_prep_w_boundaries.pkl')
        
    assign_training_data() 


def load_MESSENGER_into_tplot(date_string,res="01",full=False,FIPS=False):
    
    ''' 
    Load a single day of MESSENGER data, from the file_MESSENGER_data file
    
    Input is a string in format 'YEAR-MONTH-DAY'
    
    Output is an array of times in datetime format, 3 component magnetic field
    in the aberrated MSM' coordinate frame, magnetic field amplitude, and 3 
    component ephemeris information in the aberrated MSM' coordinate system'
    
    '''
    #res can be 01,05,10,60
    doy=get_day_of_year(date_string)
    month=date_string[5:7]
    year=date_string[2:4]
    
    year_full=date_string[0:4]
    if doy < 10: doy_s='00'+str(doy)
        
    elif (doy<100) & (doy>=10):doy_s='0'+str(doy)
        
    else: doy_s=str(doy)
    
    
    
    #file='/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated avg/MAGMSOSCIAVG'+year+str(doy)+'_'+res+'_V08.TAB'
    
    file=file_MESSENGER_data+year+'/'+month+'/'+'MAGMSOSCIAVG'+year+doy_s+'_'+res+'_V08.TAB'
    if full==True:
        file='/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated/MAGMSOSCI'+year+str(doy)+'_V08.TAB'
    df = np.genfromtxt(file)
    
    hour=df[:,2]
    
    #print(hour[0])
    
    minute=df[:,3]
    
    second=df[:,4]
    
    year=date_string[0:4]
    
    doy=int(doy_s)-1
    
 
   

    
    date=datetime.datetime(year=int(year),month=1,day=1)+datetime.timedelta(doy)
    
    #print(date)
    
    date2=[]
    
    for i in range(np.size(hour)):
        if int(hour[i])-int(hour[i-1]) < 0:
            
            doy=doy+1
            
            date=datetime.datetime(year=int(year),month=1,day=1)+datetime.timedelta(doy)
        
        date2.append(date+datetime.timedelta(hours=hour[i], minutes=minute[i], seconds=second[i]))
        
    #print(date2[0])
    
    #time=[d.strftime("%Y-%m-%d %H:%M:%S") for d in date2]
    
    #print(time[0])
    
    time=date2
    
    #time=[d.timestamp for d in date2]
    
    #return time
    
    
    #Get B
    mag1=df[:,10:13]
        
    
    
    #Get ephemeris data
    eph=df[:,7:10]
    

    
    ephx=df[:,7]
    ephy=df[:,8]
    ephz=df[:,9]
    
    #Offset due to dipole field!
    
    ephz=ephz-479
    
    #Aberration:
        
    phi=get_aberration_angle(date_string)
    
    new_magx=mag1[:,0]*np.cos(phi)-mag1[:,1]*np.sin(phi)
    
    new_magy=mag1[:,0]*np.sin(phi)+mag1[:,1]*np.cos(phi)
    
    mag1[:,0]=new_magx
    mag1[:,1]=new_magy
    
    
    new_ephx=ephx*np.cos(phi)-ephy*np.sin(phi)
    
    
    new_ephy=ephx*np.sin(phi)+ephy*np.cos(phi)
    
    ephx=new_ephx
    ephy=new_ephy
    
    eph=np.transpose(np.vstack((ephx,ephy,ephz)))
    
    if full==True:
        mag1=df[:,9:]
        eph=df[:,5:8]
        ephx=df[:,5]
        ephy=df[:,6]
        ephz=df[:,7]
    
    #Define magnetic field amplitude
    
    
    magamp=np.sqrt(mag1[:,0]**2+mag1[:,1]**2+mag1[:,2]**2)
    
    if FIPS==False:
        return time, mag1, magamp, eph
    
    if FIPS==True:
        
        year=date_string[2:4]
        
        import os
        
        
        file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V01.TAB'
        
        if os.path.isfile(file)==False:
            file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V03.TAB'
            
        if os.path.isfile(file)==False:
            file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V02.TAB'
            
        if os.path.isfile(file)==False:
            #print('No FIPS File')
            #print(file)
            return time,mag1,magamp,eph
        
        df = np.genfromtxt(file)
        
        

        erange=[13.577,12.332,11.201,10.174,9.241,8.393,7.623,6.924,
                6.289,5.712,5.188,4.713,4.28,3.888,3.531,3.207,2.913,2.646,2.403,2.183,1.983,1.801,
                1.636,1.485,1.349,1.225,1.113,1.011,0.918,0.834,0.758,0.688,0.625,0.568,
                0.516,0.468,0.426,0.386,0.351,0.319,0.29,0.263,0.239,0.217,0.197,0.179,0.163,0.148,
                0.134,0.122,0.111,0.1,0.091,0.083,0.075,0.068,0.062,0.056,0.051,0.046,
                0.046,0.046,0.046,0.046]

        df_data=df[:,2:]

        df_dates=df[:,1]
        
        cutoff=200411609
        
        #cutoff=cutoff.timestamp()

        if df_dates[-1] > cutoff:
            

            datetime_MET=convert_to_date_s('2004-08-03 06:55:32')
            
        else: 
            
            datetime_MET=convert_to_date_s('2013-01-08 20:13:20')

        df_dates2=df_dates+datetime_MET.timestamp()

        df_datetime=[convert_to_date_2_utc(d) for d in df_dates2]

        ds=df_datetime[0]

        df_datetime=[convert_to_datetime(d) for d in df_datetime]

        df_data=np.reshape(df_data,(np.size(df[:,0]),5,64))

        H_data=df_data[:,0,:]
        
        #time_FIPS=np.array(df_datetime)-datetime.timedelta(hours=1)
        
        time_FIPS=np.array(df_datetime)
        
        
        if os.path.isfile(file)==True:
        
            return time, mag1, magamp, eph, time_FIPS, H_data
            
        if os.path.isfile(file)==False:
            return time, mag1, magamp, eph                    



def downsample(new_resolution,arr,type_num=False):
    'Downsample a dataset to a new resolution'
    from scipy import stats as st
    
    new_length = len(arr) // new_resolution
    
    #Reshape the array to segments at the new_resolution length
    
    arr_reshaped = arr[:new_length * new_resolution].reshape(new_length, new_resolution)
    
    # Average the array along this axis
    
    arr_averaged= np.mean(arr_reshaped, axis=1)
    
    if type_num == True:
        
        #If downsampling the region within the magnetosphere, take the mode of the data
        arr_averaged = st.mode(arr_reshaped,axis=1)[0]
        
    
    return arr_averaged   

def generate_SW_MS_dataframe_variance(max_ang_diff):
    ''' Pre-process the data to generate the training/test set from all MESSENGER
        data
        
        This procedure requires Sun 2023 boundary crossing list and a .pkl file of
        all relavent data for MESSENGER (full_data_2.pkl)
    
    
        max_ang_diff if the maximum value of alpha, the angle between the measured
        magnetosheath vector and the reference vector just downstream of the bow shock
        (max_ang_diff=30 for the manuscript)
        
        The function generates the list of IMF targets and the list of magnetosheath
        features used to train the model
        
    '''
    import numpy as np
    import datetime
    import pandas as pd
    from trying3 import read_in_Weijie_files,convert_to_datetime,convert_to_date_2,convert_to_date_2_utc
    from scipy.stats import circmean
    from scipy.optimize import fsolve
    import logging
    logging.basicConfig(
    level=logging.DEBUG,  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',)

    # Load in data and produce crossing dataframes

    mp_in, mp_out, bs_in, bs_out = read_in_Weijie_files()
    
    #Create magnetopause inbound, outbound and bowshock inbound and outbound
    # dataframs
    
    mi=generate_crossing_dataframe(mp_in,'mpi')
    
    mo=generate_crossing_dataframe(mp_out,'mpo')
    
    bi=generate_crossing_dataframe(bs_in,'bsi')
    
    bo=generate_crossing_dataframe(bs_out,'bso')
    
    crossings=[mi,mo,bi,bo]
    
    cc=pd.concat(crossings)
    
    # Sort to create a pickle of all crossings
    c=cc.sort_values('s')
    
    
    def find_r_mp(x_mp,y_mp,z_mp):
        
        # Used for calculating magnetopause standoff distance
        rho=np.sqrt(y_mp**2+z_mp**2+x_mp**2)
        r=np.sqrt(y_mp**2+z_mp**2)
        
        phi=np.arctan2(r,x_mp)
            
        alpha=0.5
        
        Rss=rho/((2/(1+np.cos(phi)))**(alpha))
            
        return Rss
        
    def find_r_bs(x_bs,y_bs,z_bs):
        # Used for calculating bow shock standoff distance
        # Given values
        x = x_bs
        
        r_bs=np.sqrt(y_bs**2+z_bs**2)
        
        phi = np.arctan2(r_bs,x)
        
        rho_bs = np.sqrt(y_bs**2+x_bs**2+z_bs**2)
    
        # Define the equation
    
        # Define the grid ranges for each variable
        p_range = np.linspace(2.4, 2.8, 25)
        e_range = np.linspace(.8, 1.3, 25)
        X0_range = np.linspace(.3, 1.0, 25)
    
        # Perform the grid search
        best_params = None
        best_error = np.inf
    
        for p in p_range:
            for e in e_range:
                for X0 in X0_range:
                    current_params = [p, e, X0]
                    
                    psi=e
    
                    L=psi*p
    
                    phi = (np.linspace(0,2*np.pi,100))
                    rho = L/(1. + psi*np.cos(phi))
                
                    xshock = X0 + rho*np.cos(phi)
                    yshock = rho*np.sin(phi)
                    
                    d=np.sqrt((x_bs-xshock)**2+(r_bs-yshock)**2)
                    
                    current_error=np.min(d)
                    
                    if current_error < best_error:
                        best_error = current_error
                        best_params = current_params
        
        p=best_params[0]
        e=best_params[1]
        x0=best_params[2]
        
        r_bs_s=x0+e*p/(1+e)
        
        return r_bs_s
    
#     
# 
    #Downsample data to new resolution and make new downsampled dataframe
    
    # Load in the dataset 
    all_data=pd.read_pickle(save_path+'fd_prep_w_boundaries.pkl')
    
    # New resolution is 40 seconds
    nr=40
    
    #Downsample arrays in all_data to create new dataframe of eph and mag properties
    
    magx=downsample(nr,all_data.magx.to_numpy())
    magy=downsample(nr,all_data.magy.to_numpy())
    magz=downsample(nr,all_data.magz.to_numpy())
    
    magamp=downsample(nr,all_data.magamp.to_numpy())
    
    ephx=downsample(nr,all_data.ephx.to_numpy())
    ephy=downsample(nr,all_data.ephy.to_numpy())
    ephz=downsample(nr,all_data.ephz.to_numpy())
    
    type_num=downsample(nr,all_data.Type_num.to_numpy(),type_num=True)
    
    # Have to be manually compute the downsampled time array because of datetimes

    new_resolution = nr  # Number of points to average over

    time=all_data.time.to_numpy()
    
    time=np.array([pd.Timestamp(t) for t in time])
    
    
    # Calculate the new length of the arrays after averaging
    new_length = len(all_data.time.to_numpy()) // new_resolution
    
    # Reshape the arrays to prepare for averaging
    t_reshaped = time[:new_length * new_resolution].reshape(new_length, new_resolution)
    
    def tstamp(x):
        return x.timestamp()
    t_utc=np.vectorize(tstamp)(t_reshaped)
    
    # Calculate the average values
    t_averaged_utc = np.mean(t_utc, axis=1)
    
    t_averaged=np.array([convert_to_datetime(convert_to_date_2_utc(t)) for t in t_averaged_utc])
    
    all_data_40=pd.DataFrame(data={'magx':magx,'magy':magy,'magz':magz,'magamp':magamp,'time':t_averaged,'ephx':ephx,'ephy':ephy,'ephz':ephz,'Type_num':type_num[:,0]})
    
    # Save so you only have to run the downsample once
    pd.to_pickle(all_data_40,save_path+'full_data_w_boundaries_40.pkl')
# =============================================================================
    
    all_data_40=pd.read_pickle(save_path+'full_data_w_boundaries_40.pkl')
    
    # Create mag numpy file
    mag=all_data_40[['magx','magy','magz']].to_numpy()
    
    # Create ephemeris numpy file
    eph=all_data_40[['ephx','ephy','ephz']].to_numpy()
    
    magamp=all_data_40.magamp.to_numpy()
    
    time=all_data_40['time'].to_numpy()
    
    # Create solar wind dataframe for IMF targets
    
    sw=pd.DataFrame(columns=['magx','magy','magz','magamp','time','r_bs'])
    
    # Create magnetosheath dataframe for magnetosheath features
    ms=pd.DataFrame(columns=['magx','magy','magz','magamp','time','x','r','theta','ephx','ephy','ephz','r_mp','alpha'])
    
    
    x = eph[:,0]
    
    r = np.sqrt(eph[:,1]**2+eph[:,2]**2)
    
    rho_full = np.sqrt(eph[:,0]**2+eph[:,2]**2+eph[:,1]**2)
    
    phi_full = np.arctan2(r,eph[:,0])
    
    theta = np.arctan2(eph[:,1],eph[:,2])*180/np.pi
    
    type_num = all_data_40['Type_num'].to_numpy()
    
    #Track number of bow shock crossings analyzed
    bs_number_tracker = 0
    
    def angle_between_vectors(vector1, vector2):
        # Outputs angle between two vectors
        
        # Convert the input lists to arrays
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        # Dot the vectors
        dot_product = np.dot(vector1, vector2)

        # Calculate the magnitudes of the vectors
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Calculate the cosine of the angle between the vectors
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in radians
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    
    
    for i in range(np.size(c['s'])-1):
        
        # Loop through all crossings
        
        # Required buffer to make sure no information from IMF is in a magnetosheath
        # point, comes as a result of the downsampling of the data.
        edge_buffer = 20
        
        #Define the start time to the interval as the end of a crossing+edge
        start_time = c['e'].iloc[i]+datetime.timedelta(seconds=edge_buffer)
        
        
        #Define the end time of an interval as the start of the next crossing-edge
        
        end_time = c['s'].iloc[i+1]-datetime.timedelta(seconds=edge_buffer)
        
        # gd is the data within this boundary
        gd = np.where((time > start_time) & (time < end_time))[0]
        
        # What kind of crossing is the first?
        type1=c['Type'].iloc[i]
        
        # What kind of crossing is the second?
        type2=c['Type'].iloc[i+1]
        
        # Print the time
        print(c['e'].iloc[i])
        
        
        # Set time before/after BS that we average over to estimate sw magnetic field
        
        # For the manuscript, only use the most direct upstream IMF measurement (swt=1)
        swt=1
        
        # Make sure there is data within the interval loaded
        if np.size(gd)>0:
            
            if (type1=='bso' and type2=='bsi'):
                # Starting with an outbound bowshock crossing, and ending with inbound,
                # so, time in between is in the solar wind:
                
                
                
                
                if np.size(gd) > swt:
                    
                    # Calculate IMF parameters just upstream of the outbound bow shcok
                    
                    # Calculate the IMF parameters just upstream of the bow shock
                    mag_sw=np.mean(mag[gd[0:swt],:],axis=0)
                    
                    magamp_sw=np.mean(magamp[gd[0:swt]])
                    
                    eph_sw=np.mean(eph[gd[0:swt],:],axis=0)
                     
                    sw_time=pd.Timestamp(time[gd[swt//2]])
                    
                    r_bs_1=find_r_bs(eph_sw[0],eph_sw[1],eph_sw[2])
                    
                    # Add this to the solar wind dataframe
                    
                    sw.loc[i]=[mag_sw[0],mag_sw[1],mag_sw[2],magamp_sw,sw_time,r_bs_1]
                    
                    # Calculate IMF paramters just upstream of the inbound bow shock
                    
                    #Before bsi
                    
                    mag_sw=np.mean(mag[gd[-swt:],:],axis=0)
                    
                    magamp_sw=np.mean(magamp[gd[-swt:]])
                    
                    sw_time=pd.Timestamp(time[gd[-swt//2]])
                    
                    eph_sw=np.mean(eph[gd[-swt:],:],axis=0)
                    
                    r_bs_1=find_r_bs(eph_sw[0],eph_sw[1],eph_sw[2])
                    
                    # Add this to the solar wind dataframe
                    
                    sw.loc[i+1]=[mag_sw[0],mag_sw[1],mag_sw[2],magamp_sw,sw_time,r_bs_1]
                    
                    
                    


                
            if (type1=='bsi' and type2=='mpi'):
                
                # Starting with inbound bowshock and ending with inbound magnetopause,
                # so, time in between is magnetosheath
                
                # Begin the interval at the end of the bowshock crossing
                sw_time_crossing=c['e'].iloc[i]
                
                # End interval at start of magnetopause crossing
                mp_crossing=c['s'].iloc[i+1]
                
                # Make sure bow shock crossing is not greater than 15 minutes (900 seconds)
                
                t_bs=c['e'].iloc[i]+datetime.timedelta(seconds=edge_buffer)-(c['s'].iloc[i]-datetime.timedelta(seconds=edge_buffer))
                
                # Make sure bow shock crossing is not greater than 15 minutes (900 seconds)
                if t_bs.total_seconds() < 900:
                    
                    bs_number_tracker+=1
                    
                    logging.info('bs_number_tracker = '+str(bs_number_tracker))
                    
                    # Apply an edge buffer to make sure you're getting no information from across the boundary via 40 s average
                    start_time=sw_time_crossing+datetime.timedelta(seconds=edge_buffer)
                    
                    end_time=mp_crossing-datetime.timedelta(seconds=edge_buffer)
                    
                    # Define range within this time that is defined as magnetosheath data:
                    gd_ms=np.where((time >= start_time) & (time <= end_time) & (type_num==2))[0]
                    
                    if np.size(gd_ms) > 0:
                        
                        # We are starting at the bs, so reference ms point is the first ms point
                        ref_ms_measurement=gd_ms[0]
                        
                        #Find magnetic field coordinates of this point
                        bx_ms=mag[ref_ms_measurement,0]
                        by_ms=mag[ref_ms_measurement,1]
                        bz_ms=mag[ref_ms_measurement,2]
                        
                        # Chop arrays to this range
                        mag_ms_full=mag[gd_ms,:]
                        
                        eph_ms_full=eph[gd_ms,:]
                        
                        magamp_ms_full=magamp[gd_ms]
                        
                        x_ms_full=x[gd_ms]
                        
                        r_ms_full=r[gd_ms]
                        
                        theta_ms_full=theta[gd_ms]
                    
                        ms_time_full=np.array([pd.Timestamp(d) for d in time[gd_ms]])
                    
                        type_num_ms_full=type_num[gd_ms]
    
                        # Calculate angular difference between reference vector and other MS measurements
                        ang_diff=np.array([angle_between_vectors([bx_ms,by_ms,bz_ms], \
                                                                 [mag_ms_full[a,0],mag_ms_full[a,1],mag_ms_full[a,2]])\
                                           for a in range(len(mag_ms_full))])
                         
                        # Only include data within the range in which the angular difference is less than the defined max_ang_diff
                        # if ang_diff exceeds max_ang_diff, then stop appending and cut the magnetosheath data there
                        gd=[]
                        
                        p=0
                        
                        while ((ang_diff[p] < max_ang_diff) & (p<len(ang_diff))):
                            gd.append(p)
                            p+=1
                            if p>=len(ang_diff):
                                break
                        
                        gd=np.array(gd)
                    
                    
                        if np.size(gd) > 0:
                            
                            #Chop arrays again
                            
                            mag_ms=mag_ms_full[gd,:]
                            
                            eph_ms=eph_ms_full[gd,:]
                            
                            magamp_ms=magamp_ms_full[gd]
                            
                            ang_diff_ms=ang_diff[gd]
                            
                            x_ms=x_ms_full[gd]
                            
                            r_ms=r_ms_full[gd]
                            
                            theta_ms=theta_ms_full[gd]
                        
                            ms_time=np.array([pd.Timestamp(d) for d in ms_time_full[gd]])
                            
        
                            
                            
                            #find predicted radius of magnetopause for empirical prediction
                            
                            st=c['s'].iloc[i+1]-datetime.timedelta(seconds=40)
                            en=c['e'].iloc[i+1]+datetime.timedelta(seconds=40)
                            
                            
                            
                            gd_rmp=np.where((time>=st) & (time <= en))[0]
                            
                            if np.size(gd_rmp) >1:
                                
                                eph_ms_mean=np.mean(eph[gd_rmp],axis=0)
                                
                                
                                r_mp=find_r_mp(eph_ms_mean[0],eph_ms_mean[1],eph_ms_mean[2])
                            
                            
                            for j in range(len(gd)):
                                
                                # Append to ms dataframe
                                ms.loc[gd_ms[j]]=[mag_ms[j,0],mag_ms[j,1],mag_ms[j,2],magamp_ms[j],ms_time[j],x_ms[j],r_ms[j],theta_ms[j],eph_ms[j,0],eph_ms[j,1],eph_ms[j,2],r_mp,ang_diff_ms[j]]
                    
                    
                
            if (type1=='mpo' and type2 == 'bso'):
                
                # Starting with outbound magnetopause and ending with inbound bowshock,
                # so, time in between is magnetosheath
                
                # Begin the interval at the end of the magnetopause crossing
                mp_crossing=c['e'].iloc[i]
                
                # End interval at start of bowshock crossing
                bs_crossing=c['s'].iloc[i+1]
                
                # Make sure that the bow shock crossing interval is not greater that 15 minutes (900 seconds)
                
                t_bs=c['e'].iloc[i+1]+datetime.timedelta(seconds=edge_buffer)-(c['s'].iloc[i+1]-datetime.timedelta(seconds=edge_buffer))
                
                if t_bs.total_seconds() < 900:
                    
                    bs_number_tracker+=1
                    
                    logging.info('bs_number_tracker = '+str(bs_number_tracker))
                
                    # Apply an edge buffer to make sure you're getting no information from across the boundary via 40 s average
                    start_time=mp_crossing+datetime.timedelta(seconds=edge_buffer)
                    
                    end_time=bs_crossing-datetime.timedelta(seconds=edge_buffer)
                    
                    # Define range within this time that is defined as magnetosheath data:
                    gd_ms=np.where((time >= start_time) & (time <= end_time) & (type_num==2))[0]
                    
                    if np.size(gd_ms) > 0:
                        
                        # We are starting at the mp, so reference ms point is the last ms point
                        ref_ms_measurement=gd_ms[-1]
                        
                        #Find magnetic field coordinates of this point
                        bx_ms=mag[ref_ms_measurement,0]
                        by_ms=mag[ref_ms_measurement,1]
                        bz_ms=mag[ref_ms_measurement,2]
                        
                        # Chop arrays to this range
                        mag_ms_full=mag[gd_ms,:]
                        
                        eph_ms_full=eph[gd_ms,:]
                        
                        magamp_ms_full=magamp[gd_ms]
                        
                        x_ms_full=x[gd_ms]
                        
                        r_ms_full=r[gd_ms]
                        
                        theta_ms_full=theta[gd_ms]
                    
                        ms_time_full=np.array([pd.Timestamp(d) for d in time[gd_ms]])
                    
                        type_num_ms_full=type_num[gd_ms]
    
                        # Calculate angular difference between reference vector and other MS measurements
                        ang_diff=np.array([angle_between_vectors([bx_ms,by_ms,bz_ms], \
                                                                 [mag_ms_full[a,0],mag_ms_full[a,1],mag_ms_full[a,2]])\
                                           for a in range(len(mag_ms_full))])
                         
                        # Only include data within the range in which the angular difference is less than the defined max_ang_diff
                        gd=[]
                        
                        p=len(ang_diff)-1
                        
                        while ((ang_diff[p] < max_ang_diff) & (p>=0)):
                            gd.append(p)
                            p-=1
                            if p<0:
                                break
                        
                        gd=np.flip(np.array(gd))
                    
                    
                        if np.size(gd) > 0:
                            
                            #Chop arrays again
                            
                            mag_ms=mag_ms_full[gd,:]
                            
                            eph_ms=eph_ms_full[gd,:]
                            
                            ang_diff_ms=ang_diff[gd]
                            
                            magamp_ms=magamp_ms_full[gd]
                            
                            x_ms=x_ms_full[gd]
                            
                            r_ms=r_ms_full[gd]
                            
                            theta_ms=theta_ms_full[gd]
                        
                            ms_time=np.array([pd.Timestamp(d) for d in ms_time_full[gd]])
                        
                        st=c['s'].iloc[i]-datetime.timedelta(seconds=40)
                        en=c['e'].iloc[i]+datetime.timedelta(seconds=40)
                        
                        gd_rmp=np.where((time>=st) & (time <= en))[0]
                        
                        if np.size(gd_rmp) >1:
                            
                            eph_ms_mean=np.mean(eph[gd_rmp],axis=0)
                            
                            r_mp=find_r_mp(eph_ms_mean[0],eph_ms_mean[1],eph_ms_mean[2])
                            
                        for j in range(len(gd)):
                            
                            ms.loc[gd_ms[j]]=[mag_ms[j,0],mag_ms[j,1],mag_ms[j,2],magamp_ms[j],ms_time[j],x_ms[j],r_ms[j],theta_ms[j],eph_ms[j,0],eph_ms[j,1],eph_ms[j,2],r_mp,ang_diff_ms[j]]
                    
    pd.to_pickle(ms,save_path+'ms_ann_'+str(max_ang_diff)+'_diff.pkl')

    pd.to_pickle(sw,save_path+'sw_'+str(max_ang_diff)+'_diff.pkl')
    
    def generate_SW_MS_combined_dataframe_var(max_ang_diff):
        
        ''' Match IMF targets from sw dataframe to magnetosheath features'''
        
        import pandas as pd
        

        ms=pd.read_pickle(save_path+'ms_ann_'+str(max_ang_diff)+'_diff.pkl')

        sw=pd.read_pickle(save_path+'sw_'+str(max_ang_diff)+'_diff.pkl')
        
        time_ms=ms.time.to_numpy()
        
        time_sw=sw.time.to_numpy()
        
        ms=ms.reset_index()
        
        ms[['bswx','bswy','bswz','magsw','tsw','r_bs','traj']]=False
        
        for i in range(len(time_ms)):
            diff=np.abs(time_ms[i]-time_sw)
            
            gd=np.where(diff==np.min(diff))[0][0]
            
            tsw=sw.time.iloc[gd]
            
            tms=time_ms[i]
            
            if tsw>tms:
                
                traj='exiting'
                
            if tms>tsw:
                
                traj='entering'
                
            
            if i>=1:
                
                traj_prev=ms['traj'].loc[i-1]
                bswx_prev=ms['bswx'].loc[i-1]
                
                if ((traj_prev!=traj) & (bswx_prev==sw.magx.iloc[gd])):
                    breakpoint()    
            
            ms.loc[i,['bswx','bswy','bswz','magsw','tsw','r_bs','traj']]=[sw.magx.iloc[gd],sw.magy.iloc[gd],sw.magz.iloc[gd],sw.magamp.iloc[gd],sw.time.iloc[gd],sw.r_bs.iloc[gd],traj]
            
            
            
            #print(i/len(time_ms))
            
        pd.to_pickle(ms,save_path+'dataset_'+str(max_ang_diff)+'_diff.pkl')
    
    ms=generate_SW_MS_combined_dataframe_var(max_ang_diff)
    
    return ms

def create_and_train_ensemble_model_norm(num_models,filename,save_file_tag,emp_pred=True):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, BatchNormalization, Dense
    import matplotlib.pyplot as plt
    '''
    
    Code to train a specified number of artificial neural networks and make
    ensemble predictions of upstream IMF parameters for all magnetosheath data
    measured by MESSENGER
    
    num_models is the number of models used in the bagging step, generally set emp_pred to False
    
    num_models for manuscript = 100
    
    filename is the filename of the training/test dataset
    
    filename for manuscript = save_path+'dataset_30_diff'
    
    save_file_tag is the tag for the model predictions

    save_file_tag for manuscript = '30deg'
    
    Takes magnetosheath and solar wind data to create an ensemble model of IMF predictions
    
    Returns .pkl files of the newly created model run on the train/test set, and all magnetosheath data
    
    '''
    
    # filename for manuscript = save_path+'dataset_30_diff.pkl'
    
    filename=save_path+'dataset_30_diff.pkl'
    
    save_file_tag = '30deg'
    
    #Load in dataframe
    df = pd.read_pickle(filename)
    df = df.drop(columns=['traj'])
    
    if emp_pred==False:
        #Remove extreme values that aren't physical and are due to instrumental effects
        
        quartile_limit=.999
        
        quartile_threshold = df['magamp'].quantile(quartile_limit)
        
        filtered_df = df[df['magamp'] <= quartile_threshold]
        
        df=filtered_df
        
        quartile_threshold = df['magsw'].quantile(quartile_limit)
        
        filtered_df = df[df['magsw'] <= quartile_threshold]
        
        df=filtered_df
    
    # Load your data from a DataFrame (assuming df is already loaded)
    
    # Calculate heliocentric distance, and add it to the dataframe
    distance=np.array([get_mercury_distance_to_sun(t) for t in df.time])
    
    df['distance']=distance     
        
    if emp_pred==False:
        
        full_variables=['magx', 'magy', 'magz', 'magamp', 'x', 'r', 'theta','distance','ephx','ephy','ephz','r_mp','r_bs','time','tsw','alpha']
        
    
    # Define the variables that are the output/target of the model
    
    target_variables = ['bswx','bswy','bswz','magsw']
    
    # Define the variables that are the inputs to the model
    
    input_variables = ['magx', 'magy', 'magz', 'magamp', 'x', 'r', 'theta','distance']
        
    # Pull targets (y) and inputs (X) from df
    
    X = np.array(df[full_variables].values)
    y = np.array(df[target_variables].values,dtype=float)
    
    
    # This next step ensures that the data are split into training/test sets along
    # individual bowshock crossings, making all ms and IMF data from a bs crossing
    # to be put into either the training or the test set.
    
    # Manually shift data
    shift=df.shift(1)-df
    
    # Find points where the IMF target changes from one time step to the next
    transitions=np.where(shift.bswx != 0.0)[0]
    
    transitions=np.insert(transitions,len(transitions),len(shift))
    
    #Create array of bowshock segments to be separated
    bs_segments = [(transitions[i], transitions[i + 1]) for i in range(len(transitions) - 1)]
    
    int_array=np.arange(len(transitions)-1)
    
    
    # Split this list of indicies into training and test sets
    train_array_val, test_array = train_test_split(int_array, test_size=0.2, random_state=42)
    
    train_array, val_array= train_test_split(train_array_val,test_size=0.2, random_state=42)
    
    X_train=X[bs_segments[train_array[0]][0]:bs_segments[train_array[0]][1]]
    
    for j in range(len(train_array)):
        
        if j==0:
            X_train=X[bs_segments[train_array[0]][0]:bs_segments[train_array[0]][1]]
            y_train=y[bs_segments[train_array[0]][0]:bs_segments[train_array[0]][1]]
            
        else:
            X_train=np.vstack((X_train,X[bs_segments[train_array[j]][0]:bs_segments[train_array[j]][1]]))
            y_train=np.vstack((y_train,y[bs_segments[train_array[j]][0]:bs_segments[train_array[j]][1]]))
            
    for j in range(len(test_array)):
        
        if j==0:
            X_test=X[bs_segments[test_array[0]][0]:bs_segments[test_array[0]][1]]
            y_test=y[bs_segments[test_array[0]][0]:bs_segments[test_array[0]][1]]
            
        else:
            X_test=np.vstack((X_test,X[bs_segments[test_array[j]][0]:bs_segments[test_array[j]][1]]))
            y_test=np.vstack((y_test,y[bs_segments[test_array[j]][0]:bs_segments[test_array[j]][1]]))
    for j in range(len(val_array)):
        
        if j==0:
            X_val=X[bs_segments[val_array[0]][0]:bs_segments[val_array[0]][1]]
            y_val=y[bs_segments[val_array[0]][0]:bs_segments[val_array[0]][1]]
            
        else:
            X_val=np.vstack((X_val,X[bs_segments[val_array[j]][0]:bs_segments[val_array[j]][1]]))
            y_val=np.vstack((y_val,y[bs_segments[val_array[j]][0]:bs_segments[val_array[j]][1]]))
    
    # Create new variables that only have relavant magnetosheath quantities
        
    X_train_all=X_train
    X_test_all=X_test
    X_val_all=X_val
    
    X_all=X
    
    X_train_run=X_train[:,0:8]
    X_test_run=X_test[:,0:8]
    X_val_run=X_val[:,0:8]
    
    X_run=np.array(X[:,0:8])
    
    # Create an empty list to store individual models
    models = []
    
    np.random.seed(30)
    # Create a list of seeds to make the same random choice of training set each time
    
    seed_array=np.arange(100)
    
    seeds=np.random.choice(seed_array,size=len(seed_array),replace=False)
    for i in range(num_models):
        # Create a new model for each bootstrap iteration
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Input, Dense, BatchNormalization, Lambda
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        
        # Assuming X_train_run and X_val_run are your input features
        # Assuming y_train and y_val are your target labels
        
        # Normalize your input features, setting them from 0 to 1
        scaler_X = MinMaxScaler()
        X_train_normalized = scaler_X.fit_transform(X_train_run)
        X_val_normalized = scaler_X.transform(X_val_run)
        
        # Normalize your targets, setting them from 0 to 1
        scaler_y = MinMaxScaler()
        y_train_normalized = scaler_y.fit_transform(y_train)
        y_val_normalized = scaler_y.transform(y_val)
        
        # Begin bagging regression
        
        # First, define an array of indicies to be randomized
        index_array=np.arange(len(X_train_normalized))
        
        # Randomize the indicies
        np.random.seed(seeds[i])
        sample_array=np.random.choice(index_array,size=len(index_array), replace=True
                                      )
        
        # Only include the randomized portion of the training set
        X_train_rs=X_train_normalized[sample_array,:]
        
        y_train_rs=y_train_normalized[sample_array,:]
        
        
        if num_models < 50:
            
            # Only use the bagging regression for high number of model runs
            # If using a small number of runs, then just use the full training set
            
            X_train_rs=X_train_normalized
            
            y_train_rs=y_train_normalized
            
        
        # Build the model with normalization and denormalization layers
        model = Sequential([
            Input(shape=(8,)),
            BatchNormalization(),  # Input normalization layer
        
            Dense(24, activation='tanh'),
            BatchNormalization(),  # Hidden layer normalization
            Dense(24, activation='tanh'),
            BatchNormalization(),  # Hidden layer normalization
            Dense(24, activation='tanh'),
            BatchNormalization(),  # Hidden layer normalization
        
            Dense(4, activation='linear'),
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model with normalized data
        history = model.fit(X_train_rs, y_train_rs, validation_data=(X_val_normalized, y_val_normalized), epochs=50, batch_size=32)
        
        # Optionally, if you want to use the original y_train and y_val for evaluation:
        
        models.append(model)
        
        
        
        
        #breakpoint()
        
        if ((i==0) & (num_models < 50)):
            
            import shap
            
            model.save(save_path+'shap_test_model.h5')
    
            # Train the model with normalized data
            #history = model.fit(X_train_rs, y_train_rs, validation_data=(X_val_normalized, y_val_normalized), epochs=50, batch_size=32)
            
            # Initialize an explainer object
            explainer = shap.Explainer(model, X_train_rs)
            
            input_variables_shap=['BX','BY','BZ','|B|',"$X_{MSM'}$",'r','Theta','$R_{HC}$']
            
            X_val_df=pd.DataFrame(data=X_train_normalized,columns=input_variables_shap)
            
            np.save(save_path+'X_train_rs_shap.npy',X_train_rs)
            
            np.save(save_path+'X_test_rs_shap.npy',X_test)
            X_val_df.to_pickle(save_path+'X_train_shap.pkl')
            
            #breakpoint()
            # # Calculate Shapley values for the validation data
            # shap_values = explainer(X_val_df)
            
            
            # # Summarize the effects of all the features
            # shap.summary_plot(shap_values, scaler_X.inverse_transform(X_val_normalized), feature_names=df[input_variables].columns)
        
        

    
    def create_predictions(X_array,df_model_run):
        
        predictions = np.zeros_like(X_array[:,0:4], dtype=np.float32)
        
        for model in models:
            predictions += scaler_y.inverse_transform(model.predict(X_array))
        
        model_predictions=np.zeros((num_models, len(X_array), 4))
        for i, model in enumerate(models):
            # Predictions from the model
            print(i)
            prediction = model.predict(X_array)
            
            prediction = scaler_y.inverse_transform(prediction)
            
            model_predictions[i, :, :] = prediction
            
        ensemble_predictions = predictions / num_models
        uncertainty = np.std(model_predictions, axis=0)
        
        df_model_run[['bswx_pred','bswy_pred','bswz_pred','magsw_pred']]=ensemble_predictions
        
        df_model_run[['bswx_err','bswy_err','bswz_err','magsw_err']]=uncertainty
        
        np.save('model_predictions.npy',model_predictions)
        
        return df_model_run
    
    
    save_dir = 'saved_models'

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, model in enumerate(model):
        model.save(save_path+save_dir+"/model_{i}.h5")
    
    df_test=pd.DataFrame(data=X_test_all,columns=full_variables)
    
    df_test[target_variables]=y_test
    
    X_test_normalized=scaler_X.transform(X_test_run)
    
    df_test=create_predictions(X_test_normalized,df_test)
    
    
    
    
    df_train=pd.DataFrame(data=X_train_all,columns=full_variables)
    
    df_train[['bswx','bswy','bswz','magsw']]=y_train
    #X_test_normalized=scaler.transform(X_test)
    
    X_train_normalized=scaler_X.transform(X_train_run)
    
    
    df_train=create_predictions(X_train_normalized,df_train)
    
    
    
    df_full=pd.DataFrame(data=X_all,columns=full_variables)
    
    X_run_normalized=scaler_X.transform(X_run)

    df_full=create_predictions(X_run_normalized,df_full)

        
    df_full[['bswx','bswy','bswz','magsw','tsw','ephx','ephy','ephz']]=df[['bswx','bswy','bswz','magsw','tsw','ephx','ephy','ephz']]
    
    pd.to_pickle(df_test,save_path+'df_attempt_TEST_ensemble_'+str(num_models)+'_models_'+save_file_tag+'.pkl')
    
    pd.to_pickle(df_full,save_path+'df_attempt_FULL_ensemble_'+str(num_models)+'_models_norm_'+save_file_tag+'.pkl')
    
    pd.to_pickle(df_train,save_path+'df_attempt_TRAIN_ensemble_'+str(num_models)+'_models_norm_'+save_file_tag+'.pkl')
    
    
    all_data_40=pd.read_pickle(save_path+'full_data_w_boundaries_40.pkl')
    

    
    ms=all_data_40[all_data_40.Type_num==2]
    
    distance_ms=np.array([get_mercury_distance_to_sun(t) for t in ms.time])
    
    ms['distance']=distance_ms
    
    eph=ms[['ephx','ephy','ephz']].to_numpy()
    
    x=eph[:,0]
    
    r=np.sqrt(eph[:,1]**2+eph[:,2]**2)
    
    theta=np.arctan2(eph[:,1],eph[:,2])*180/np.pi
    
    ms['x']=x
    ms['r']=r
    ms['theta']=theta
    
    
    
    X_all_2 = np.array(ms[input_variables],dtype=float)
    
    X_all_2 = scaler_X.transform(X_all_2)
    
    ms=create_predictions(X_all_2,ms)
    
    pd.to_pickle(ms,save_path+'df_attempt_ALL_MS_ensemble_'+str(num_models)+'_models_norm_'+save_file_tag+'.pkl')
    
def feature_distro(save=False):
    '''Generate feature distribution plot (Figure 2 in the manuscript)'''
    
    fs=25
    lw=6
    import numpy as np
    #Data Used for training/test data:
    #df=pd.read_pickle('dataset_10_min_lim_'+str(40)+'.pkl')
    #max_ang_diff=30
    #df=pd.read_pickle('dataset_'+str(max_ang_diff)+'_diff.pkl')
    
    df=pd.read_pickle(save_path+'full_data_w_boundaries_40.pkl')
    df['x']=df.ephx
    
    r=np.sqrt(df.ephy**2+df.ephz**2)
    
    theta=np.arctan2(df.ephy,df.ephz)*180/np.pi
    
    df['r']=r
    df['theta']=theta
    
    
    
    ms=df[df.Type_num==2]
    sw=df[df.Type_num==3]
    
    #SW_Properties
    bswx=sw.magx.to_numpy()
    bswy=sw.magy.to_numpy()
    bswz=sw.magz.to_numpy()
    
    magsw=sw.magamp.to_numpy()
    
    import numpy as np
    ca_sw=np.array([np.arctan2(bswy[i],bswz[i])*180/np.pi for i in range(len(bswy))])
    ca_ms=np.arctan2(ms.magy,ms.magz)*180/np.pi
    
    cone_sw=np.array([np.arccos(bswx[i]/magsw[i])*180/np.pi for i in range(len(bswy))])
    cone_ms=np.arccos(ms.magx/ms.magamp)*180/np.pi
    
    sw[['ca_sw','cone_sw']]=np.transpose(np.vstack((ca_sw,ca_sw)))
    ms[['ca_ms','cone_ms']]=np.transpose(np.vstack((ca_ms,cone_ms)))
    
    def hist_creation_2(sw_ar,ms_ar,rang,titl,xtitl,save=False):
        
        '''Create 2 histograms comparing solar wind to magnetosheath data'''
        
        fig,ax=plt.subplots(1)
        
        counts,bins=np.histogram(sw_ar,bins=30,range=rang)
        
        ax.hist(bins[:-1],bins,weights=counts/np.size(sw_ar),color='gold',histtype='step',linewidth=lw,label='IMF')
        
        #ax.hist(sw_ar,bins=30,range=rang,density=True,color='gold',histtype='step',linewidth=3,label='IMF_Measured')
        
        counts,bins=np.histogram(ms_ar,bins=30,range=rang)
        
        ax.hist(bins[:-1],bins,weights=counts/np.size(ms_ar),color='royalblue',histtype='step',linewidth=lw,label='Magnetosheath')

        ax.tick_params(axis='y',labelsize=fs-4)
        ax.tick_params(axis='x',labelsize=fs-4)
        

        ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
            
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=4)
        #ax.hist(ms_ar,bins=30,range=rang,density=True,color='royalblue',histtype='step',linewidth=3,label='MS_Measured')
        
        ax.legend(fontsize=fs)
        
        ax.set_xticks(np.arange(rang[0],rang[1]+1,round((rang[1]-rang[0])/6)))
        
        ax.set_title(titl,fontsize=fs)
        
        ax.set_xlabel(xtitl,fontsize=fs)
        
        ax.set_ylabel('Normalized Frequency',fontsize=fs)
        
        fname=titl
        if save==True:
            plt.savefig('/Users/bowersch/Desktop/DIAS/Work/Papers/ANN_modeling/Figures/'+fname+'.png',dpi=600)
        
        
    def feature_hist_creation(ar1,ar2,rang,titl,xtitl,label1,label2,save=False):
        
        fig,ax=plt.subplots(1)
        
        counts,bins=np.histogram(ar1,bins=30,range=rang)
        
        ax.hist(bins[:-1],bins,weights=counts/np.size(ar1),color='indianred',histtype='step',linewidth=lw,label=label1)
        
        #ax.hist(sw_ar,bins=30,range=rang,density=True,color='gold',histtype='step',linewidth=3,label='IMF_Measured')
        
        counts,bins=np.histogram(ar2,bins=30,range=rang)
        
        ax.hist(bins[:-1],bins,weights=counts/np.size(ar2),color='mediumturquoise',histtype='step',linewidth=lw,label=label2)

    
        #ax.hist(ms_ar,bins=30,range=rang,density=True,color='royalblue',histtype='step',linewidth=3,label='MS_Measured')
        
        ax.legend(fontsize=fs-4)
        ax.tick_params(axis='y',labelsize=fs-4)
        ax.tick_params(axis='x',labelsize=fs-4)
        

        ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
            
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=4)
        
        ax.set_xticks(np.arange(rang[0],rang[1]+1,round((rang[1]-rang[0])/6)))
        
        ax.set_title(titl,fontsize=fs)
        
        ax.set_xlabel(xtitl,fontsize=fs)
        
        ax.set_ylabel('Normalized Frequency',fontsize=fs)
        
        fname=titl
        if save==True:
            plt.savefig('/Users/bowersch/Desktop/DIAS/Work/Papers/ANN_modeling/Figures/'+fname+'.png',dpi=600)
        
        

    
    hist_creation_2(ca_sw,ca_ms,[-180,180],'Clock Angle Distribution','Clock Angle (deg)')
    
        
    hist_creation_2(cone_sw,cone_ms,[0,180],'Cone Angle Distribution','Cone Angle (deg)')
    hist_creation_2(magsw,ms.magamp,[0,120],'|B| Distribution','|B| (nT)')
    
    
    
    plt.rcParams.update({'font.size': 11})
     
     
     
    df_day=ms[(ms.r >2.5)]
    df_night=ms[(ms.r<2.5)]
    
    df_st=ms[(np.abs(ms.theta) < 90)]
    df_lt=ms[(np.abs(ms.theta) > 90)]
    
    df_px=ms[(ms.ephx >0)]
    df_nx=ms[(ms.ephx<0)]
    
    from trying3 import get_mercury_distance_to_sun
    
    distance=np.array([get_mercury_distance_to_sun(t) for t in df.time])
    
    df['distance']=distance
    
    df_aph=df[(df.distance > .45)]
    
    df_per=df[(df.distance < .32)]
            
    feature_hist_creation(df_day.magamp,df_night.magamp,[0,120],'|B| with r','|B| (nT)','r>2.5 $R_{M}$','r<2.5 $R_{M}$')

    feature_hist_creation(df_lt.cone_ms,df_st.cone_ms,[0,180],'Cone Angle with $\Theta$','Cone Angle (deg)','|$\Theta$| > 90$^\circ$','|$\Theta$| < 90$^\circ$')

    feature_hist_creation(df_px.ca_ms,df_nx.ca_ms,[-180,180],"Clock Angle with $X_{MSM'}$",'Clock Angle (deg)',"$X_{MSM'}$>0","$X_{MSM'}$<0")

     
    feature_hist_creation(df_aph.magamp,df_per.magamp,[0,120],'|B| with Heliocentric Distance','|B| (nT)','Aphelion ($r_{hc}$ > .45 AU)','Perihelion ($r_{hc}$ <.32 AU)')
 
    
def r2_analysis():
    ''' Generate the r2 analysis plots for each output, assessing model performance
    on the test set
    
    This makes Figure 5 in the manuscript
    
    '''
    
    df=pd.read_pickle(save_path+'df_attempt_TEST_ensemble_100_models_30deg.pkl')
    
    
    bswx=np.array([float(i) for i in df.bswx])
    bswx_pred=np.array([float(i) for i in df.bswx_pred])
    bswx_emp=np.array([float(i) for i in df.bx_emp_pred])
    
    bswy=np.array([float(i) for i in df.bswy])
    bswy_pred=np.array([float(i) for i in df.bswy_pred])
    bswy_emp=np.array([float(i) for i in df.by_emp_pred])
    
    bswz=np.array([float(i) for i in df.bswz])
    bswz_pred=np.array([float(i) for i in df.bswz_pred])
    bswz_emp=np.array([float(i) for i in df.bz_emp_pred])
    
    magsw=np.array([float(i) for i in df.magsw])
    magsw_pred=np.array([float(i) for i in df.magsw_pred])
    magsw_emp=np.sqrt(bswx_emp**2+bswy_emp**2+bswz_emp**2)
    
    r2_bx=r2_score(df.bswx,df.bswx_pred)
    
    r2_by=r2_score(df.bswy,df.bswy_pred)
    
    r2_bz=r2_score(df.bswz,df.bswz_pred)
    
    r2_magamp=r2_score(df.magsw,df.magsw_pred)
    
    rmean=np.mean([r2_bx,r2_by,r2_bz,r2_magamp])
    
    print(f'R2 BX : {r2_bx:.4f}')
    print(f'R2 BY : {r2_by:.4f}')
    print(f'R2 BZ : {r2_bz:.4f}')
    print(f'R2 |B| : {r2_magamp:.4f}')
    
    print(f'R2 mean : {rmean:.4f}')
    
    
    fig,ax=plt.subplots(2,2)

    
    colors=['blue','green','red','black']
    
    
    def format_plot(ax,meas,pred,color,r2,lab,min_val,max_val):
        si=.3
        
        fs=18
        
        ax.scatter(meas,pred,s=si,color=color)
        ax.set_xlabel(lab+' Measured (nT)',fontsize=fs)
        ax.set_ylabel(lab+' Predicted (nT)',fontsize=fs)
        ax.set_aspect('equal')
    
        a,b= np.polyfit(meas, pred, 1)
    
        x=np.arange(-100,100,step=.1)
        ax.plot(x,x,color='grey',label=f'$r^2$={r2:.2f}')
        
        ax.text(0.02, 0.93, f'$r^2$={r2:.2f}', transform=ax.transAxes, color='black', fontsize=20, va='top', ha='left')
        #ax.legend()
    
        ax.set_xlim(min_val,max_val)
        ax.set_ylim(min_val,max_val)
        ax.tick_params(labelsize=fs-4)
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=4)
        
        
    
   
    format_plot(ax[0,0],df.bswx,df.bswx_pred,colors[0],r2_bx,'IMF $B_X$',-60,60)
    format_plot(ax[1,0],df.bswy,df.bswy_pred,colors[1],r2_by,'IMF $B_Y$',-60,60)
    format_plot(ax[0,1],df.bswz,df.bswz_pred,colors[2],r2_bz,'IMF $B_Z$',-60,60)
    format_plot(ax[1,1],df.magsw,df.magsw_pred,colors[3],r2_magamp,'IMF $|B|$',0,100)

        
        
    fig,ax=plt.subplots(2,2)
    df['magsw_emp_pred']=magsw_emp
    r2_bx=r2_score(df.bswx,df.bx_emp_pred)
    
    r2_by=r2_score(df.bswy,df.by_emp_pred)
    
    r2_bz=r2_score(df.bswz,df.bz_emp_pred)
    
    
    
    r2_magamp=r2_score(magsw,magsw_emp)
    format_plot(ax[0,0],df.bswx,df.bx_emp_pred,colors[0],r2_bx,'IMF $B_X$',-60,60)
    format_plot(ax[1,0],df.bswy,df.by_emp_pred,colors[1],r2_by,'IMF $B_Y$',-60,60)
    format_plot(ax[0,1],df.bswz,df.bz_emp_pred,colors[2],r2_bz,'IMF $B_Z$',-60,60)
    format_plot(ax[1,1],df.magsw,df.magsw_emp_pred,colors[3],r2_magamp,'IMF $|B|$',0,100)
    

    
    rmean=np.mean([r2_bx,r2_by,r2_bz,r2_magamp])
    print("Empirical Model:")
    print(f'R2 BX : {r2_bx:.4f}')
    print(f'R2 BY : {r2_by:.4f}')
    print(f'R2 BZ : {r2_bz:.4f}')
    print(f'R2 |B| : {r2_magamp:.4f}')
    
    print(f'R2 mean : {rmean:.4f}') 
    

def r2_analysis_exploration():
    ''' Plot to see how the r2 score changes with certain parameters
    
    Using the test set, how does r2 change with parameters like alpha and strength
    of magnetosheath magnetic fields
    
    This forms Figure 6 from the manuscript
    
    '''
    
    npl=12
    #df=pd.read_pickle('df_attempt_TEST_'+str(npl)+'.pkl')
    #df=pd.read_pickle('df_attempt_TEST_ensemble_50_models.pkl')
    
    #filename_test='/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/df_attempt_TEST_ensemble_2_models_shap_test.pkl'
    
    filename_full = save_path+'/df_attempt_FULL_ensemble'\
        '_100_models_norm_30deg.pkl'
    filename_all=save_path + 'df_attempt_ALL_MS_ensemble_100_models_norm_30deg.pkl'
    filename_test=save_path+'df_attempt_TEST_ensemble_100_models_30deg.pkl'
    filename_train=save_path+'df_attempt_TRAIN_ensemble_100_models_norm_30deg.pkl'
    
    df = pd.read_pickle(filename_test)
    
    diff = df['tsw']-df['time']
    
    diff=np.array([np.abs(d.total_seconds()) for d in diff])/60.
    
    X = np.array(df['x'].values,dtype=float)
    r = np.array(df['r'].values,dtype=float)
    
    
    rho=np.sqrt(X**2+r**2)
    
    df['rho']=rho
    
    time_range=np.arange(2,120,(120-2)/15)
    
    alpha_range=np.arange(0,33,(33-0)/15)
    
    time_range_bx=np.arange(15,80,(80-15)/15)
    
    time_range_amp=np.arange(10,100,(100-10)/15)
    
    r_range=np.arange(1.5,5,(5-1.5)/15.)
    
    def create_r2_dep_plot(param_x,param_y,param_z,param_mag,t_range):
        
    
        r2_all=np.array([])
        r2_bx_all=np.array([])
        r2_by_all=np.array([])
        r2_bz_all=np.array([])
        r2_magamp_all=np.array([])
        
        
        bx_error=np.array([])
        by_error=np.array([])
        bz_error=np.array([])
        magsw_error=np.array([])
    
        
        
        for i in range(len(t_range)-1):
            max_time=t_range[i+1]
            min_time=t_range[i]
            
            gd_x=np.where((param_x >= min_time) & (param_x < max_time))
            gd_y=np.where((param_y >= min_time) & (param_y < max_time))
            gd_z=np.where((param_z >= min_time) & (param_z < max_time))
            gd_mag=np.where((param_mag >= min_time) & (param_mag < max_time))
            
            
            r2_bx=r2_score(df.bswx.iloc[gd_x],df.bswx_pred.iloc[gd_x])
            
            r2_by=r2_score(df.bswy.iloc[gd_y],df.bswy_pred.iloc[gd_y])
            
            r2_bz=r2_score(df.bswz.iloc[gd_z],df.bswz_pred.iloc[gd_z])
            
            r2_magamp=r2_score(df.magsw.iloc[gd_mag],df.magsw_pred.iloc[gd_mag])
            
            rmean=np.mean([r2_bx,r2_by,r2_bz,r2_magamp])
            
            r2_bx_all=np.append(r2_bx_all,r2_bx)
            r2_by_all=np.append(r2_by_all,r2_by)
            r2_bz_all=np.append(r2_bz_all,r2_bz)
            
            r2_magamp_all=np.append(r2_magamp_all,r2_magamp)
            
            r2_all=np.append(r2_all,rmean)
            
            magsw_error=np.append(magsw_error,np.mean(df['magsw_err'].iloc[gd_mag]))
            bx_error=np.append(bx_error,np.mean(df['bswx_err'].iloc[gd_x]))
            by_error=np.append(by_error,np.mean(df['bswy_err'].iloc[gd_y]))
            bz_error=np.append(bz_error,np.mean(df['bswz_err'].iloc[gd_z]))
        
        
        return r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,bx_error,by_error,bz_error,magsw_error
    
    
    
    def format_plot(ar_1,ar_2,ar_3,ar_4,ar_5,x_label,y_label,t_range):
        fig,ax=plt.subplots(1)
        lw=4.0
        si=55
    
        #ax.scatter(t_range[0:-1],ar_1,color='blue',s=si)
        ax.plot(t_range[0:-1],ar_1,color='blue',linewidth=lw,label='IMF BX',alpha=.8)
        
        #ax.scatter(t_range[0:-1],ar_2,color='green',s=si)
        ax.plot(t_range[0:-1],ar_2,color='green',linewidth=lw,label='IMF BY',alpha=.8)
        
        #ax.scatter(t_range[0:-1],ar_3,color='red',s=si)
        ax.plot(t_range[0:-1],ar_3,color='red',linewidth=lw,label='IMF BZ',alpha=.8)
        
        #ax.scatter(t_range[0:-1],ar_4,color='black',s=si)
        ax.plot(t_range[0:-1],ar_4,color='black',linewidth=lw,label='IMF |B|',alpha=.8)
        
        #ax.scatter(t_range[0:-1],ar_5,color='grey',s=si)
        ax.plot(t_range[0:-1],ar_5,color='grey',linewidth=lw,label='Mean',alpha=.8)
    
        fs=18
        
        ax.legend(fontsize=fs, loc='lower right', bbox_to_anchor=(1.5, 0.0))
       
        ax.tick_params(labelsize=fs-4)
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=4)
        
        ax.set_xlabel(x_label,fontsize=fs)
        ax.set_ylabel(y_label,fontsize=fs)
        ax.set_ylim(.2,1)
        
        
        
    r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,bx_error,by_error,bz_error,magsw_error=create_r2_dep_plot(diff,diff,diff,diff,time_range)
    
    format_plot(r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,'Minutes from Solar Wind','$r^2$ Score',time_range)
    
    
    
    #format_plot(bx_error,by_error,bz_error,magsw_error,np.mean(np.vstack((bx_error,by_error,bz_error,magsw_error)),axis=0),'Minutes from Solar Wind','Prediction Error',time_range)
    
    
    
    r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,bx_error,by_error,bz_error,magsw_error=\
        create_r2_dep_plot(np.abs(df.magx),np.abs(df.magy),np.abs(df.magz),np.abs(df.magamp),time_range_bx)
    
    format_plot(r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,'$|B_X|$, $|B_Y|$, $|B_Z|$, $|B|$ (nT)','$r^2$ Score',time_range_bx)
    
    
    
    r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,bx_error,by_error,bz_error,magsw_error=\
        create_r2_dep_plot(np.abs(df.magamp),np.abs(df.magamp),np.abs(df.magamp),np.abs(df.magamp),time_range_amp)
    
    format_plot(r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,'Magnitude (nT)','$r^2$ Score',time_range_amp)
    
    
    r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,bx_error,by_error,bz_error,magsw_error=\
        create_r2_dep_plot(df.alpha,df.alpha,df.alpha,df.alpha,alpha_range)
    
    format_plot(r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,'\u03B1 (deg)','$r^2$ Score',alpha_range)
    
    r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,bx_error,by_error,bz_error,magsw_error=\
        create_r2_dep_plot(df.rho,df.r,df.rho,df.rho,r_range)
    
    format_plot(r2_bx_all,r2_by_all,r2_bz_all,r2_magamp_all,r2_all,'r ($R_{M}$)','$r^2$ Score',r_range)
    
    

def plot_region_time_series_plots(trange):
    
    
    ''' Generate magnetic field and orbital trajectory plot for a time range
    colored by the associated region of the magnetosphere as determined from the
    Sun2023 list (Figure 1 in the manuscript)'''
    
    trange2=[['2012-09-13 02:22:10','2012-09-13 10:38:27']]
    
    ax=plot_MESSENGER_trange_3ax(trange2[0],plot=False)
    fig,ax2=plt.subplots(len(trange2))
    fs=18
    ps=15
    
    for p in range(len(trange2)):
        trange=trange2[p]
        
        # start at the start of the trange
        start=convert_to_datetime(trange[0])
        
        # End at the end
        end=convert_to_datetime(trange[1])
        
        ad=pd.read_pickle(save_path+'/fd_prep_w_boundaries.pkl')
        
        ad_full=ad
        
        ad=ad[((ad.time > start) & (ad.time < end))]
        
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # Magnetosphere portion
        ms=ad[(ad['Type_num']==1)]
        
        # Magnetosheeath portion
        sheath=ad[(ad['Type_num']==2)]
        
        # Solar Wind portion
        sw=ad[(ad['Type_num']==3)]
        
        # Bow Shock Portion
        bs=ad[(ad['Type_num']==4)]
        
        # Magnetopause portion
        mp=ad[(ad['Type_num']==5)]

        
        locations=[ms,mp,sheath,bs,sw]
        
        # Colors for different regions
        colors=['firebrick','mediumpurple','royalblue','orange','gold']
        
        # Labels 
        labelss=['Magnetosphere','Magnetopause','Magnetosheath','Bowshock','Solar Wind']
        
        for i in range(len(locations)):
            
        
            ax[0].scatter(locations[i].ephx,locations[i].ephz-.19,color=colors[i],s=ps,label=labelss[i])
            ax[1].scatter(locations[i].ephx,locations[i].ephy,color=colors[i],s=ps,label=labelss[i])
            ax[2].scatter(locations[i].ephy,locations[i].ephz-.19,color=colors[i],s=ps,label=labelss[i])
            
        
        #a=pd.read_pickle('fd_prep_w_boundaries.pkl')
        
        n=np.array([np.size(np.where(ad_full['Type_num']==1)),np.size(np.where(ad_full['Type_num']==5)),\
           np.size(np.where(ad_full['Type_num']==2)),np.size(np.where(ad_full['Type_num']==4)),\
            np.size(np.where(ad_full['Type_num']==3))])/len(ad_full)
        
        
        def generate_c_plot(trange,ax_k=False):
            #Generate the colored plot
            fs=18
            start_date=convert_to_datetime(trange[0])
            
            end_date=convert_to_datetime(trange[1])
            
            ad=pd.read_pickle(save_path+'fd_prep_w_boundaries.pkl')
            
            ad=ad[(ad.time > start_date) & (ad.time < end_date)]
            
            diff=ad.shift(1)-ad
            
            transitions=np.where(diff.Type_num != 0.0)[0]
            
            transistions=np.insert(transitions,0,0)
            
            transitions=np.insert(transitions,len(transitions),len(diff)-1)
            
            
            if ax_k==False:
                fig,ax=plt.subplots(1)
            
            else: ax=ax_k
            
            colors=['firebrick','royalblue','gold','orange','mediumpurple']
            labels=['Magnetosphere','Magnetosheath','Solar Wind','Bowshock','Magnetopause']
            for i in range(len(transitions)-1):
                
                plt.rcParams.update({'font.size': 12})
                
                si=transitions[i]
                fi=transitions[i+1]
                

                ax.plot(ad.time.iloc[si:fi],ad.magamp.iloc[si:fi],color=colors[round(np.nanmean(ad.Type_num.iloc[si:fi]))-1],label=labels[round(np.nanmean(ad.Type_num.iloc[si:fi]))-1])
                
                ax.set_ylabel('|B| (nT)',fontsize=18)
                
                ax.tick_params(labelsize=fs-4)
                
                ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                ax.tick_params(axis='both', which='major', length=8)
                ax.tick_params(axis='both', which='minor', length=4)
            
                ax.set_yscale('log')   
        def plot_region_loop(i,df,start_time,stop_time,color):
            ax2[i].scatter(df.time[((df.time > start_time) & (df.time < stop_time))],df.magamp[((df.time > start_time) & (df.time < stop_time))],color=color,s=ps)
            
            ax2[i].set_ylabel('$|B|$',fontsize=fs)

        if len(trange2)>1:  
            generate_c_plot(trange,ax_k=ax2[p])
            ax2[len(trange2)-1].set_xlabel('Time',fontsize=fs)
            
        else:
            generate_c_plot(trange,ax_k=ax2)
            ax2.set_xlabel(trange[0][0:10],fontsize=fs)
            
            import matplotlib.dates as mdates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


def create_all_meshes_ANN():
    '''Create all coverage maps as viewed from different planes around Mercury
    
    Run this after the model is created and the dataset is created
    
    Creates Figure 4 in the manusc
    
    '''
    
    
    
    filename_full=save_path+'df_attempt_FULL_ensemble_100_models_norm_30deg.pkl'
    filename_all=save_path+'df_attempt_ALL_MS_ensemble_100_models_norm_30deg.pkl'
    filename_test=save_path+'df_attempt_TEST_ensemble_100_models_30deg.pkl'
    
    
    def magnetosheath_mesh_plot_YZ(filename_full,filename_all,Bepi=False):
        
    
        import pandas as pd

        import datetime

        import numpy as np

        from trying3 import convert_to_datetime,convert_datetime_to_string, load_MESSENGER_into_tplot,read_in_Weijie_files, check_for_mp_bs_WS,plot_mp_and_bs
        
        #Load in a pandas dataframe with all of the lobe magnetic field data and position
        
        def create_meshes(Lobe_Data):

            sc=50
            #Create_Meshes
            
            sc2=50
            
            mesh_full=np.zeros((sc,1,sc,3))
            
            count_full=np.zeros((sc,1,sc,3))
            
            mesh_eph=np.zeros((sc,1,sc,3))
           
            count_eph=np.zeros((sc,1,sc,3))
           
           
            R_m=2440
           
            distance=18.0
           
            pos_y=np.arange(sc)*distance/(sc)-distance/2.0
           
            distance=18.0
           
            pos_z=np.arange(sc2)*distance/(sc2)-distance/2.0
           
            pos_x=[-10,10]
            #mesh goes r, theta, z, mag
           
            mesh_mag=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1,3))
            
            count=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1))
            
            mesh_eph=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1,3))
            
            
            R_m=2440
           
            
            ephx=Lobe_Data.ephx
            ephy=Lobe_Data.ephy
            ephz=Lobe_Data.ephz
            
            theta_d=np.arctan2(ephy,ephx)
                
            r=np.sqrt(ephy**2+ephx**2)
            
            ex=ephx
            
            ey=ephy
            
            ez=ephz
            
            magx=Lobe_Data.magx
            
            
            magy=Lobe_Data.magy
            
            magz=Lobe_Data.magz
            
            magamp=Lobe_Data.magamp
           
            r_pos_x=np.roll(pos_x,-1)
           
            r_pos_y=np.roll(pos_y,-1)
           
            r_pos_z=np.roll(pos_z,-1)
           
            for rr in range(np.size(pos_x)):
                print(rr)
           
                for tt in range(np.size(pos_y)):
                
                    for zz in range(np.size(pos_z)):
                    
                        gd_a=np.where((ex > pos_x[rr]) & (ex < r_pos_x[rr]) &\
                                    (ey > pos_y[tt]) & (ey < r_pos_y[tt]) &\
                                        (ez > pos_z[zz]) & (ez < r_pos_z[zz]))[0]
                        
                        if np.size(gd_a) > 0:
                            
                            l=gd_a[0]
                            
                                                            
                            x=rr
                            y=tt
                            z=zz
               
                                
                            magx_gd=magx.iloc[gd_a]
                            
                            magy_gd=magy.iloc[gd_a]
                            
                            magz_gd=magz.iloc[gd_a]
                            
                            ephx_gd=ephx.iloc[gd_a]
                            
                            ephy_gd=ephy.iloc[gd_a]
                            
                            ephz_gd=ephz.iloc[gd_a]
                            
                            magamp_gd=magamp.iloc[gd_a]
                            
                            mesh_mag[x,y,z,0]=np.mean(magx_gd)
                            
                            
                            mesh_mag[x,y,z,1]=np.mean(magy_gd)
                            
                            mesh_mag[x,y,z,2]=np.mean(magz_gd)
                            
                            count[x,y,z]=np.size(magz_gd)
                            
                            mesh_eph[x,y,z,0]=np.mean(ephx_gd)
                            
                            mesh_eph[x,y,z,1]=np.mean(ephy_gd)
                            
                            mesh_eph[x,y,z,2]=np.mean(ephz_gd)
                            
                            
            return count,pos_x,pos_y,pos_z
            
        #s_40=pd.read_pickle('dataset_10_min_lim_40w_pred.pkl')
        #full=pd.read_pickle('ms_ann_full_'+str(40)+'.pkl')

        
        all_ms=pd.read_pickle(filename_all)
        
        full_tt=pd.read_pickle(filename_full)
        
        
        #Same Heliocentric Distance
        
        #all_ms=all_ms[((all_ms.distance >.39) & (all_ms.distance <.40))]
        
        #full_tt=full_tt[((full_tt.distance >.39) & (full_tt.distance <.40))]
        
        count_lobe,pos_x,pos_y,pos_z=create_meshes(full_tt)
        
        count_tail,pos_x,pos_y,pos_z=create_meshes(all_ms)
            
                           
        def format_eph_plot(axis):
            fs=18
            #axis.fill_between([10,-10],[-10,-10],color='gainsboro')
            
            #axis.fill_between([10,-10],[10,10],color='gainsboro')
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)-.19
            
            axis.plot(x, y, color='black')
            
            axis.set_aspect('equal',adjustable='box')
        
            # X component
            xlim=[-5,5]
        
            ylim=[-8,2]
        
            
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            
            
            axis.set_xlabel("$Y_{MSM'}$",fontsize=fs)
            
            axis.set_ylabel("$Z_{MSM'}$",fontsize=fs)
            
            axis.tick_params(labelsize=fs-4)
            
            axis.fill_between(x, y, where=x<0, color='black', interpolate=True)
            axis.fill_between(x, y, where=x<0,color='black',interpolate=True)
            
            axis.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            axis.yaxis.set_minor_locator(AutoMinorLocator())
            
            axis.tick_params(axis='both', which='major', length=8)
            axis.tick_params(axis='both', which='minor', length=4)
            
            
            

            
        fig, ax=plt.subplots(1,3)
        

        
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        
        #newcolors[:1,:]= white
        newcmp = ListedColormap(newcolors)
        
        viridis = cm.get_cmap('plasma', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        newcmp_r = ListedColormap(newcolors)
        
        count_lobe=count_lobe[0,:-2,:-2]
        
        count_tail=count_tail[0,:-2,:-2]
        
        ratio=count_lobe/count_tail
        
        ratio[count_tail<0]=float('nan')
        
        
        
        count_lobe[count_tail<1]=float('nan')
        
        ratio[((count_lobe==0) & (count_tail > 0))]=0
        
        count_lobe[((count_lobe==0) & (count_tail > 0))]=0
        
        
        
        count_tail[count_tail<1]=float('nan')
        

        
        format_eph_plot(ax[0])
        format_eph_plot(ax[1])
        format_eph_plot(ax[2])
        
        
        fig1=ax[0].pcolormesh(pos_y,pos_z,np.transpose(count_lobe/1000.),shading='flat',cmap=newcmp)
        
        fig2=ax[1].pcolormesh(pos_y,pos_z,np.transpose(count_tail/1000.),shading='flat',cmap='viridis')
        
        fig3=ax[2].pcolormesh(pos_y,pos_z,np.transpose(ratio),shading='flat',cmap=newcmp_r)
        
        ax[0].set_title('Train/Test Magnetosheath Measurements')
        
        ax[1].set_title('All Magnetosheath Measurements')
        
        ax[2].set_title('Ratio')
        
        fig.colorbar(fig1,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig2,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig3,label='Ratio',fraction=0.046, pad=0.09,orientation="horizontal") 
             
        if Bepi==True:
            
            def plot_orbit_traj(apogee,perigee,ax,c,l,reverse=False):
                # Constants
                
                lw=3.3
                # Semi-major axis
                a = (apogee + perigee) / 2
                
                # Eccentricity
                e = (apogee - perigee) / (apogee + perigee)
                
                # Function to calculate radius from angle theta
                def radius(theta):
                    return a * (1 - e**2) / (1 + e * np.cos(theta))
                
                # Generate values for theta
                theta_values = np.linspace(0, 2*np.pi, 1000)
                
                # Calculate radius values
                r_values = radius(theta_values)
                if reverse==True:
                    r_values=-radius(theta_values)
                
                
                # Convert polar coordinates to Cartesian coordinates
                x_values = r_values * np.cos(theta_values)
                y_values = r_values * np.sin(theta_values)-.2
                
                ax.plot(x_values,y_values,linewidth=lw,color=c,label=l)
            
            plot_orbit_traj(1.6,1.2,ax[0],'hotpink','MPO Autumn')
            plot_orbit_traj(1.6,1.2,ax[0],'red','MPO Spring',reverse=True)
            
            plot_orbit_traj(5.78,1.24,ax[0],'aquamarine','Mio Autumn')
            plot_orbit_traj(5.78,1.24,ax[0],'lime','Mio Spring',reverse=True)
            ax[0].legend()
            ax[0].set_ylim(-3.5,3.5)
            ax[0].set_xlim(-3.5,3.5)
            
                
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)*2.01
            y = np.sin(theta)*2.01-0.2
        
            ax[0].plot(x, y, color='black',linewidth=3,linestyle='--')
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)*3.3
            y = np.sin(theta)*3.3-0.2
        
            ax[0].plot(x, y, color='black',linewidth=3,linestyle='--')
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)-0.2
        
            ax[0].plot(x, y, color='black',linewidth=3)
    
        
    def magnetosheath_mesh_plot_XY(filename_full,filename_all):
        import pandas as pd

        import datetime

        import numpy as np

        from trying3 import convert_to_datetime,convert_datetime_to_string, load_MESSENGER_into_tplot,read_in_Weijie_files, check_for_mp_bs_WS,plot_mp_and_bs
        
        #Load in a pandas dataframe with all of the lobe magnetic field data and position
        
        def create_meshes(Lobe_Data):

            sc=50
            #Create_Meshes
        
            sc2=50
            
            mesh_full=np.zeros((sc,1,sc,3))
            
            count_full=np.zeros((sc,1,sc,3))
            
            mesh_eph=np.zeros((sc,1,sc,3))
           
            count_eph=np.zeros((sc,1,sc,3))
           
           
            R_m=2440
           
            distance=20.0
           
            pos_y=np.arange(sc)*distance/(sc)-distance/2.0
           
            distance=20.0
           
            pos_x=np.arange(sc2)*distance/(sc2)-distance/2.0
           
            pos_z=[-10,10]
            #mesh goes r, theta, z, mag
           
            mesh_mag=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))
            
            count=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0]))
            
            mesh_eph=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))
            
            
            R_m=2440
           
            
            ephx=Lobe_Data.ephx
            ephy=Lobe_Data.ephy
            ephz=Lobe_Data.ephz
            
            theta_d=np.arctan2(ephy,ephx)
                
            r=np.sqrt(ephy**2+ephx**2)
            
            ex=ephx
            
            ey=ephy
            
            ez=ephz
            
            magx=Lobe_Data.magx
            
            
            magy=Lobe_Data.magy
            
            magz=Lobe_Data.magz
            
            #magamp=np.sqrt(magx**2+magy**2+magz**2)
            
            magamp=Lobe_Data.magamp
           
            r_pos_x=np.roll(pos_x,-1)
           
            r_pos_y=np.roll(pos_y,-1)
           
            r_pos_z=np.roll(pos_z,-1)
           
            for rr in range(np.size(pos_x)):
                print(rr)
           
                for tt in range(np.size(pos_y)):
                
                    for zz in range(np.size(pos_z)):
                    
                        gd_a=np.where((ex > pos_x[rr]) & (ex < r_pos_x[rr]) &\
                                    (ey > pos_y[tt]) & (ey < r_pos_y[tt]) &\
                                        (ez > pos_z[zz]) & (ez < r_pos_z[zz]))[0]
                        
                        if np.size(gd_a) > 0:
                            
                            l=gd_a[0]
                            
                                                            
                            x=rr
                            y=tt
                            z=zz
               
                                
                            magx_gd=magx.iloc[gd_a]
                            
                            magy_gd=magy.iloc[gd_a]
                            
                            magz_gd=magz.iloc[gd_a]
                            
                            ephx_gd=ephx.iloc[gd_a]
                            
                            ephy_gd=ephy.iloc[gd_a]
                            
                            ephz_gd=ephz.iloc[gd_a]
                            
                            magamp_gd=magamp.iloc[gd_a]
                            
                            mesh_mag[x,y,z,0]=np.mean(magx_gd)
                            
                            
                            mesh_mag[x,y,z,1]=np.mean(magy_gd)
                            
                            mesh_mag[x,y,z,2]=np.mean(magz_gd)
                            
                            count[x,y,z]=np.size(magz_gd)
                            
                            mesh_eph[x,y,z,0]=np.mean(ephx_gd)
                            
                            mesh_eph[x,y,z,1]=np.mean(ephy_gd)
                            
                            mesh_eph[x,y,z,2]=np.mean(ephz_gd)
                            
                            
            return count,pos_x,pos_y,pos_z
            
        #s_40=pd.read_pickle('dataset_10_min_lim_40w_pred.pkl')
        #full=pd.read_pickle('ms_ann_full_'+str(40)+'.pkl')

        
        all_ms=pd.read_pickle(filename_all)
        
        full_tt=pd.read_pickle(filename_full)
        
        
        #Same Heliocentric Distance
        
        #all_ms=all_ms[((all_ms.distance >.39) & (all_ms.distance <.40))]
        
        #full_tt=full_tt[((full_tt.distance >.39) & (full_tt.distance <.40))]
        
        count_lobe,pos_x,pos_y,pos_z=create_meshes(full_tt)
        
        count_tail,pos_x,pos_y,pos_z=create_meshes(all_ms)
            
                           
        def format_eph_plot(axis):
            fs=18
            #axis.fill_between([10,-10],[-10,-10],color='gainsboro')
            
            #axis.fill_between([10,-10],[10,10],color='gainsboro')
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)
            
            axis.plot(x, y, color='black')
            
            axis.set_aspect('equal',adjustable='box')
        
            # X component
            xlim=[-5,3]
        
            ylim=[-5,5]
        
            
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            
            
            axis.set_xlabel("$X_{MSM'}$",fontsize=fs)
            
            axis.set_ylabel("$Y_{MSM'}$",fontsize=fs)
            
            axis.tick_params(labelsize=fs-4)
            
            axis.fill_between(x, y, where=x<0, color='black', interpolate=True)
            axis.fill_between(x, y, where=x<0,color='black',interpolate=True)
            
            axis.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            axis.yaxis.set_minor_locator(AutoMinorLocator())
            
            axis.tick_params(axis='both', which='major', length=8)
            axis.tick_params(axis='both', which='minor', length=4)
            
            plot_mp_and_bs(axis)
            
            
            

            
        fig, ax=plt.subplots(1,3)
        

        
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        
        #newcolors[:1,:]= white
        newcmp = ListedColormap(newcolors)
        
        viridis = cm.get_cmap('plasma', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        newcmp_r = ListedColormap(newcolors)
        
        count_lobe=count_lobe[:-2,:-2,0]
        
        count_tail=count_tail[:-2,:-2,0]
        
        ratio=count_lobe/count_tail
        
        ratio[count_tail<0]=float('nan')
        
        
        
        count_lobe[count_tail<1]=float('nan')
        
        ratio[((count_lobe==0) & (count_tail > 0))]=0
        
        count_lobe[((count_lobe==0) & (count_tail > 0))]=0
        
        
        
        count_tail[count_tail<1]=float('nan')
        

        
        format_eph_plot(ax[0])
        format_eph_plot(ax[1])
        format_eph_plot(ax[2])
        
        
        fig1=ax[0].pcolormesh(pos_x,pos_y,np.transpose(count_lobe/1000.),shading='flat',cmap=newcmp)
        
        fig2=ax[1].pcolormesh(pos_x,pos_y,np.transpose(count_tail/1000.),shading='flat',cmap='viridis')
        
        fig3=ax[2].pcolormesh(pos_x,pos_y,np.transpose(ratio),shading='flat',cmap=newcmp_r)
        
        ax[0].set_title('Train/Test Magnetosheath Measurements')
        
        ax[1].set_title('All Magnetosheath Measurements')
        
        ax[2].set_title('Ratio')
        
        fig.colorbar(fig1,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig2,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig3,label='Ratio',fraction=0.046, pad=0.09,orientation="horizontal") 
        
    def magnetosheath_mesh_plot_XZ(filename_full,filename_all,Bepi=False):
        import pandas as pd

        import datetime

        import numpy as np

        from trying3 import convert_to_datetime,convert_datetime_to_string, load_MESSENGER_into_tplot,read_in_Weijie_files, check_for_mp_bs_WS,plot_mp_and_bs
        
        #Load in a pandas dataframe with all of the lobe magnetic field data and position
        
        def create_meshes(Lobe_Data):

            sc=50
            #Create_Meshes
            
            sc2=50
            
            mesh_full=np.zeros((sc,1,sc,3))
            
            count_full=np.zeros((sc,1,sc,3))
            
            mesh_eph=np.zeros((sc,1,sc,3))
           
            count_eph=np.zeros((sc,1,sc,3))
           
           
            R_m=2440
           
            distance=18.0
           
            pos_x=np.arange(sc)*distance/(sc)-distance/2.0
           
            distance=18.0
           
            pos_z=np.arange(sc)*distance/(sc)-distance/2.0
           
            pos_y=[-10,10]
            #mesh goes r, theta, z, mag
           
            mesh_mag=np.zeros((sc+1,np.shape(pos_y)[0],sc2+1,3))
            
            count=np.zeros((sc+1,np.shape(pos_y)[0],sc2+1))
            
            mesh_eph=np.zeros((sc+1,np.shape(pos_y)[0],sc2+1,3))
            
            
            R_m=2440
           
            
            ephx=Lobe_Data.ephx
            ephy=Lobe_Data.ephy
            ephz=Lobe_Data.ephz
            
            theta_d=np.arctan2(ephy,ephx)
                
            r=np.sqrt(ephy**2+ephx**2)
            
            ex=ephx
            
            ey=ephy
            
            ez=ephz
            
            magx=Lobe_Data.magx
            
            
            magy=Lobe_Data.magy
            
            magz=Lobe_Data.magz
            
            magamp=Lobe_Data.magamp
           
            r_pos_x=np.roll(pos_x,-1)
           
            r_pos_y=np.roll(pos_y,-1)
           
            r_pos_z=np.roll(pos_z,-1)
           
            for rr in range(np.size(pos_x)):
                print(rr)
           
                for tt in range(np.size(pos_y)):
                
                    for zz in range(np.size(pos_z)):
                    
                        gd_a=np.where((ex > pos_x[rr]) & (ex < r_pos_x[rr]) &\
                                    (ey > pos_y[tt]) & (ey < r_pos_y[tt]) &\
                                        (ez > pos_z[zz]) & (ez < r_pos_z[zz]))[0]
                        
                        if np.size(gd_a) > 0:
                            
                            l=gd_a[0]
                            
                                                            
                            x=rr
                            y=tt
                            z=zz
               
                                
                            magx_gd=magx.iloc[gd_a]
                            
                            magy_gd=magy.iloc[gd_a]
                            
                            magz_gd=magz.iloc[gd_a]
                            
                            ephx_gd=ephx.iloc[gd_a]
                            
                            ephy_gd=ephy.iloc[gd_a]
                            
                            ephz_gd=ephz.iloc[gd_a]
                            
                            magamp_gd=magamp.iloc[gd_a]
                            
                            mesh_mag[x,y,z,0]=np.mean(magx_gd)
                            
                            
                            mesh_mag[x,y,z,1]=np.mean(magy_gd)
                            
                            mesh_mag[x,y,z,2]=np.mean(magz_gd)
                            
                            count[x,y,z]=np.size(magz_gd)
                            
                            mesh_eph[x,y,z,0]=np.mean(ephx_gd)
                            
                            mesh_eph[x,y,z,1]=np.mean(ephy_gd)
                            
                            mesh_eph[x,y,z,2]=np.mean(ephz_gd)
                            
                            
            return count,pos_x,pos_y,pos_z
            
        #s_40=pd.read_pickle('dataset_10_min_lim_40w_pred.pkl')
        #full=pd.read_pickle('ms_ann_full_'+str(40)+'.pkl')

        
        all_ms=pd.read_pickle(filename_all)
        
        full_tt=pd.read_pickle(filename_full)
        
        
        #Same Heliocentric Distance
        
        #all_ms=all_ms[((all_ms.distance >.39) & (all_ms.distance <.40))]
        
        #full_tt=full_tt[((full_tt.distance >.39) & (full_tt.distance <.40))]
        
        count_lobe,pos_x,pos_y,pos_z=create_meshes(full_tt)
        
        count_tail,pos_x,pos_y,pos_z=create_meshes(all_ms)
            
                           
        def format_eph_plot(axis):
            fs=18
            #axis.fill_between([10,-10],[-10,-10],color='gainsboro')
            
            #axis.fill_between([10,-10],[10,10],color='gainsboro')
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)-.19
            
            axis.plot(x, y, color='black')
            
            axis.set_aspect('equal',adjustable='box')
        
            # X component
            xlim=[-5,5]
        
            ylim=[-8,2]
        
            
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            
            
            axis.set_xlabel("$X_{MSM'}$",fontsize=fs)
            
            axis.set_ylabel("$Z_{MSM'}$",fontsize=fs)
            
            axis.tick_params(labelsize=fs-4)
            
            axis.fill_between(x, y, where=x<0, color='black', interpolate=True)
            axis.fill_between(x, y, where=x<0,color='black',interpolate=True)
            
            axis.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            axis.yaxis.set_minor_locator(AutoMinorLocator())
            
            axis.tick_params(axis='both', which='major', length=8)
            axis.tick_params(axis='both', which='minor', length=4)
            
            
            plot_mp_and_bs(axis)

            
        fig, ax=plt.subplots(1,3)
        

        
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        
        #newcolors[:1,:]= white
        newcmp = ListedColormap(newcolors)
        
        viridis = cm.get_cmap('plasma', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        newcmp_r = ListedColormap(newcolors)
        
        count_lobe=count_lobe[:-2,0,:-2]
        
        count_tail=count_tail[:-2,0,:-2]
        
        ratio=count_lobe/count_tail
        
        ratio[count_tail<0]=float('nan')
        
        
        
        count_lobe[count_tail<1]=float('nan')
        
        ratio[((count_lobe==0) & (count_tail > 0))]=0
        
        count_lobe[((count_lobe==0) & (count_tail > 0))]=0
        
        
        
        count_tail[count_tail<1]=float('nan')
        

        
        format_eph_plot(ax[0])
        format_eph_plot(ax[1])
        format_eph_plot(ax[2])
        
        
        fig1=ax[0].pcolormesh(pos_x,pos_z,np.transpose(count_lobe/1000.),shading='flat',cmap=newcmp)
        
        fig2=ax[1].pcolormesh(pos_x,pos_z,np.transpose(count_tail/1000.),shading='flat',cmap='viridis')
        
        fig3=ax[2].pcolormesh(pos_x,pos_z,np.transpose(ratio),shading='flat',cmap=newcmp_r)
        
        ax[0].set_title('Train/Test Magnetosheath Measurements')
        
        ax[1].set_title('All Magnetosheath Measurements')
        
        ax[2].set_title('Ratio')
        
        fig.colorbar(fig1,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig2,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig3,label='Ratio',fraction=0.046, pad=0.09,orientation="horizontal")
        
        if Bepi==True:
            
            def plot_orbit_traj(apogee,perigee,ax,c,l,reverse=False):
                # Constants
                
                lw=3.3
                # Semi-major axis
                a = (apogee + perigee) / 2
                
                # Eccentricity
                e = (apogee - perigee) / (apogee + perigee)
                
                # Function to calculate radius from angle theta
                def radius(theta):
                    return a * (1 - e**2) / (1 + e * np.cos(theta))
                
                # Generate values for theta
                theta_values = np.linspace(0, 2*np.pi, 1000)
                
                # Calculate radius values
                r_values = radius(theta_values)
                if reverse==True:
                    r_values=-radius(theta_values)
                
                
                # Convert polar coordinates to Cartesian coordinates
                x_values = r_values * np.cos(theta_values)
                y_values = r_values * np.sin(theta_values)-.2
                
                ax.plot(x_values,y_values,linewidth=lw,color=c,label=l)
            
            plot_orbit_traj(1.6,1.2,ax[0],'hotpink','MPO Winter')
            plot_orbit_traj(1.6,1.2,ax[0],'red','MPO Summer',reverse=True)
            
            plot_orbit_traj(5.78,1.24,ax[0],'aquamarine','Mio Winter')
            plot_orbit_traj(5.78,1.24,ax[0],'lime','Mio Summer',reverse=True)
            ax[0].legend()
            ax[0].set_ylim(-3.5,3.5)
            ax[0].set_xlim(-3.5,3.5)
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)-0.2
        
            ax[0].plot(x, y, color='black',linewidth=3)
        
        import pandas as pd

        import datetime

        import numpy as np

        from trying3 import convert_to_datetime,convert_datetime_to_string, load_MESSENGER_into_tplot,read_in_Weijie_files, check_for_mp_bs_WS,plot_mp_and_bs
        
        #Load in a pandas dataframe with all of the lobe magnetic field data and position
        
        def create_meshes(Lobe_Data):

            sc=60
            #Create_Meshes
            
            sc2=60
            
            mesh_full=np.zeros((sc,1,sc,3))
            
            count_full=np.zeros((sc,1,sc,3))
            
            mesh_eph=np.zeros((sc,1,sc,3))
           
            count_eph=np.zeros((sc,1,sc,3))
           
           
            R_m=2440
           
            distance=18.0
           
            pos_y=np.arange(sc)*distance/(sc)-distance/2.0
           
            distance=9.0
           
            pos_z=np.arange(sc2)*distance/(sc2)-distance/1.2
           
            pos_x=[-10,10]
            #mesh goes r, theta, z, mag
           
            mesh_mag=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1,3))
            
            count=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1))
            
            mesh_eph=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1,3))
            
            mesh_magamp=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1))
            
            mesh_distance=np.zeros((np.shape(pos_x)[0],sc+1,sc2+1))
            
            
            
            R_m=2440
           
            
            ephx=Lobe_Data.ephx
            ephy=Lobe_Data.ephy
            ephz=Lobe_Data.ephz
            
            theta_d=np.arctan2(ephy,ephx)
                
            r=np.sqrt(ephy**2+ephx**2)
            
            ex=ephx
            
            ey=ephy
            
            ez=ephz
            
            magx=Lobe_Data.bswx_pred
            
            
            magy=Lobe_Data.bswy_pred
            
            magz=Lobe_Data.bswz_pred
            
            magamp=Lobe_Data.magsw_pred
            
            distance=Lobe_Data.distance
           
            r_pos_x=np.roll(pos_x,-1)
           
            r_pos_y=np.roll(pos_y,-1)
           
            r_pos_z=np.roll(pos_z,-1)
           
            for rr in range(np.size(pos_x)):
                print(rr)
           
                for tt in range(np.size(pos_y)):
                
                    for zz in range(np.size(pos_z)):
                    
                        gd_a=np.where((ex > pos_x[rr]) & (ex < r_pos_x[rr]) &\
                                    (ey > pos_y[tt]) & (ey < r_pos_y[tt]) &\
                                        (ez > pos_z[zz]) & (ez < r_pos_z[zz]))[0]
                        
                        if np.size(gd_a) > 0:
                            
                            l=gd_a[0]
                            
                                                            
                            x=rr
                            y=tt
                            z=zz
               
                                
                            magx_gd=magx.iloc[gd_a]
                            
                            magy_gd=magy.iloc[gd_a]
                            
                            magz_gd=magz.iloc[gd_a]
                            
                            ephx_gd=ephx.iloc[gd_a]
                            
                            ephy_gd=ephy.iloc[gd_a]
                            
                            ephz_gd=ephz.iloc[gd_a]
                            
                            magamp_gd=magamp.iloc[gd_a]
                            
                            distance_gd=distance.iloc[gd_a]
                            
                            mesh_mag[x,y,z,0]=np.mean(magx_gd)
                            
                            
                            mesh_mag[x,y,z,1]=np.mean(magy_gd)
                            
                            mesh_mag[x,y,z,2]=np.mean(magz_gd)
                            
                            count[x,y,z]=np.size(magz_gd)
                            
                            mesh_magamp[x,y,z]=np.mean(magamp_gd)
                            
                            mesh_distance[x,y,z]=np.mean(distance_gd)
                            
                            
                            mesh_eph[x,y,z,0]=np.mean(ephx_gd)
                            
                            mesh_eph[x,y,z,1]=np.mean(ephy_gd)
                            
                            mesh_eph[x,y,z,2]=np.mean(ephz_gd)
                            
                            
                            
                            
            return count,mesh_mag,mesh_magamp,mesh_distance,pos_x,pos_y,pos_z
            
        
        
        full_tt=pd.read_pickle(filename_all)
        
        
        #Same Heliocentric Distance
        
        #full_tt=full_tt[((full_tt.distance >.39) & (full_tt.distance <.40))]
        
        
        count_lobe,mag_lobe,magamp_lobe,distance_lobe,pos_x,pos_y,pos_z=create_meshes(full_tt)
        
        #count_tail,pos_x,pos_y,pos_z=create_meshes(full)
            
                           
        def format_eph_plot(axis):
            fs=18
            #axis.fill_between([10,-10],[-10,-10],color='gainsboro')
            
            #axis.fill_between([10,-10],[10,10],color='gainsboro')
            
            theta = np.linspace(0, 2*np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)-.19
            
            axis.plot(x, y, color='black')
            
            axis.set_aspect('equal',adjustable='box')
        
            # X component
            xlim=[-5,5]
        
            ylim=[-8,2]
        
            
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            
            
            axis.set_xlabel("$Y_{MSM'}$",fontsize=fs)
            
            axis.set_ylabel("$Z_{MSM'}$",fontsize=fs)
            
            axis.tick_params(labelsize=fs-4)
            
            axis.fill_between(x, y, where=x<0, color='black', interpolate=True)
            axis.fill_between(x, y, where=x<0,color='black',interpolate=True)
            
            axis.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            axis.yaxis.set_minor_locator(AutoMinorLocator())
            
            axis.tick_params(axis='both', which='major', length=8)
            axis.tick_params(axis='both', which='minor', length=4)
            
            
            

            
        fig, ax=plt.subplots(1,3)
        

        
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        viridis = cm.get_cmap('bwr', 1000)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([.1])
        newcolors[500, :] = pink
        newcmp = ListedColormap(newcolors)
        
        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        
        #newcolors[:1,:]= white
        newcmp_v = ListedColormap(newcolors)
        
        viridis = cm.get_cmap('plasma', 256)
        newcolors = viridis(np.linspace(0, 1, 1000))
        pink = np.array([212/256, 212/256, 212/256,1])
        
        white=np.array([255/256, 255/256, 255/256,1])
        newcolors[:1, :] = pink
        
        #newcolors[:1,:]= white
        newcmp_p = ListedColormap(newcolors)
        
        count_lobe=count_lobe[0,:-2,:-2]
        
        mag_lobe=mag_lobe[0,:-2,:-2,:]
        
        # ratio=count_lobe/count_tail
        
        # ratio[count_tail<0]=float('nan')
        
        mx=mag_lobe[:,:,0]
        my=mag_lobe[:,:,1]
        mz=mag_lobe[:,:,2]
        
        magamp=magamp_lobe[0,:-2,:-2]
        
        distance=distance_lobe[0,:-2,:-2]
        
        mx[count_lobe<1]=0
        my[count_lobe<1]=0
        mz[count_lobe<1]=0
        
        magamp[count_lobe<1]=0
        
        distance[count_lobe<1]=0
        
        #magamp=np.sqrt(mx**2+my**2+mz**2)
        
        # ratio[((count_lobe==0) & (count_tail > 0))]=0
        
        # count_lobe[((count_lobe==0) & (count_tail > 0))]=0
        
        
        
        # count_tail[count_tail<1]=float('nan')
        
        

        
        format_eph_plot(ax[0])
        format_eph_plot(ax[1])
        format_eph_plot(ax[2])
        
        fig,ax3=plt.subplots(1,2)
        format_eph_plot(ax3[0])
        format_eph_plot(ax3[1])
        
        
        fig1=ax[0].pcolormesh(pos_y,pos_z,np.transpose(mx),shading='flat',cmap=newcmp,vmin=-40,vmax=40)
        
        fig2=ax[1].pcolormesh(pos_y,pos_z,np.transpose(my),shading='flat',cmap=newcmp,vmin=-20,vmax=20)
        
        fig3=ax[2].pcolormesh(pos_y,pos_z,np.transpose(mz),shading='flat',cmap=newcmp,vmin=-20,vmax=20)
        
        fig4=ax3[0].pcolormesh(pos_y,pos_z,np.transpose(magamp),shading='flat',cmap=newcmp_v,vmin=8,vmax=50)
        
        fig5=ax3[1].pcolormesh(pos_y,pos_z,np.transpose(distance),shading='flat',cmap=newcmp_p,vmin=.3,vmax=.47)
        
        ax[0].set_title('BX')
        
        ax[1].set_title('BY')
        
        ax[2].set_title('BZ')
        
        ax3[0].set_title('Predicted IMF |B|',fontsize=20)
        
        ax3[1].set_title('Heliocentric Distance',fontsize=20)
        
        fig.colorbar(fig1,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig2,label='Seconds '+r'$\times$ $10^3$',fraction=0.046, pad=0.09,orientation="horizontal")
        fig.colorbar(fig3,label='Ratio',fraction=0.046, pad=0.09,orientation="horizontal")    

        cbar=fig.colorbar(fig4,label='|B|',fraction=0.046, pad=0.13,orientation="horizontal")

        cbar.ax.tick_params(axis='x',labelsize=15)
        
        cbar.set_label('|B| (nT)',fontsize=20)
        
        cbar2=fig.colorbar(fig5,fraction=0.046, pad=0.13,orientation="horizontal")

        cbar2.ax.tick_params(axis='x',labelsize=15)
        
        cbar2.set_label('$r_H$ (AU)',fontsize=20)
        
        
        npl=12
        #df=pd.read_pickle('df_attempt_TEST_'+str(npl)+'.pkl')
        #df=pd.read_pickle('df_attempt_TEST_ensemble_50_models.pkl')
        df=pd.read_pickle(filename_test)
        
        
        bswx=np.array([float(i) for i in df.bswx])
        bswx_pred=np.array([float(i) for i in df.bswx_pred])
        
        
        bswy=np.array([float(i) for i in df.bswy])
        bswy_pred=np.array([float(i) for i in df.bswy_pred])
        
        
        bswz=np.array([float(i) for i in df.bswz])
        bswz_pred=np.array([float(i) for i in df.bswz_pred])
        
        
        magsw=np.array([float(i) for i in df.magsw])
        magsw_pred=np.array([float(i) for i in df.magsw_pred])
       
        def format_plot(ax,meas,pred,color,r2,lab,min_val,max_val):
            si=.3
            
            fs=18
            
            ax.scatter(meas,pred,s=si,color=color)
            ax.set_xlabel(lab+' Measured (nT)',fontsize=fs)
            ax.set_ylabel(lab+' Predicted (nT)',fontsize=fs)
            ax.set_aspect('equal')
        
            a,b= np.polyfit(meas, pred, 1)
        
            x=np.arange(-100,100,step=.1)
            ax.plot(x,x,color='grey',label=f'$r^2$={r2:.2f}')
            
            ax.text(0.02, 0.93, f'$r^2$={r2:.2f}', transform=ax.transAxes, color='black', fontsize=20, va='top', ha='left')
            #ax.legend()
        
            ax.set_xlim(min_val,max_val)
            ax.set_ylim(min_val,max_val)
            ax.tick_params(labelsize=fs-4)
            
            ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            ax.tick_params(axis='both', which='major', length=8)
            ax.tick_params(axis='both', which='minor', length=4)
        
        if emp_pred==True:
            bswx_emp=np.array([float(i) for i in df.bx_emp_pred])
            bswy_emp=np.array([float(i) for i in df.by_emp_pred])
            bswz_emp=np.array([float(i) for i in df.bz_emp_pred])
            
            magsw_emp=np.sqrt(bswx_emp**2+bswy_emp**2+bswz_emp**2)
            
            r2_bx=r2_score(df.bswx,df.bx_emp_pred)
            
            r2_by=r2_score(df.bswy,df.by_emp_pred)
            
            r2_bz=r2_score(df.bswz,df.bz_emp_pred)
            colors=['blue','green','red','black']
            fig,ax=plt.subplots(2,2)
            df['magsw_emp_pred']=magsw_emp
            
            r2_magamp=r2_score(magsw,magsw_emp)
            format_plot(ax[0,0],df.bswx,df.bx_emp_pred,colors[0],r2_bx,'IMF $B_X$',-60,60)
            format_plot(ax[1,0],df.bswy,df.by_emp_pred,colors[1],r2_by,'IMF $B_Y$',-60,60)
            format_plot(ax[0,1],df.bswz,df.bz_emp_pred,colors[2],r2_bz,'IMF $B_Z$',-60,60)
            format_plot(ax[1,1],df.magsw,df.magsw_emp_pred,colors[3],r2_magamp,'IMF $|B|$',0,100)
            

            
            rmean=np.mean([r2_bx,r2_by,r2_bz,r2_magamp])
            print("Empirical Model:")
            print(f'R2 BX : {r2_bx:.4f}')
            print(f'R2 BY : {r2_by:.4f}')
            print(f'R2 BZ : {r2_bz:.4f}')
            print(f'R2 |B| : {r2_magamp:.4f}')
            
            print(f'R2 mean : {rmean:.4f}')
            
        
            
        colors=['blue','green','red','black']
        r2_bx=r2_score(df.bswx,df.bswx_pred)
        
        r2_by=r2_score(df.bswy,df.bswy_pred)
        
        r2_bz=r2_score(df.bswz,df.bswz_pred)
        
        r2_magamp=r2_score(df.magsw,df.magsw_pred)
        
        rmean=np.mean([r2_bx,r2_by,r2_bz,r2_magamp])
        
        print(f'R2 BX : {r2_bx:.4f}')
        print(f'R2 BY : {r2_by:.4f}')
        print(f'R2 BZ : {r2_bz:.4f}')
        print(f'R2 |B| : {r2_magamp:.4f}')
        
        print(f'R2 mean : {rmean:.4f}')
        
        
        fig,ax=plt.subplots(2,2)

        
       
        
        

            
            
        
       
        format_plot(ax[0,0],df.bswx,df.bswx_pred,colors[0],r2_bx,'IMF $B_X$',-60,60)
        format_plot(ax[1,0],df.bswy,df.bswy_pred,colors[1],r2_by,'IMF $B_Y$',-60,60)
        format_plot(ax[0,1],df.bswz,df.bswz_pred,colors[2],r2_bz,'IMF $B_Z$',-60,60)
        format_plot(ax[1,1],df.magsw,df.magsw_pred,colors[3],r2_magamp,'IMF $|B|$',0,100)
    
    magnetosheath_mesh_plot_XY(filename_full, filename_all)
    
    magnetosheath_mesh_plot_YZ(filename_full, filename_all,Bepi=True)
    
    magnetosheath_mesh_plot_XZ(filename_full, filename_all,Bepi=True)
    
def shap_analysis():
    '''Generate the SHAP analysis plot from the train dataset formed from creating 
    a version of the model with fewer than 50 models
    
    Generates Figure 7 of the manuscript
    '''
    import shap
    
    from tensorflow.keras.models import load_model
    
    model=load_model(save_path+'test_model.h5')
    
    X_val_df=pd.read_pickle('X_val_shap.pkl')
    
    X_train_rs=np.load('X_train_rs_shap.npy')
    
    input_variables_shap=['BX','BY','BZ','|B|',"$X_{MSM'}$",'r','Theta','$R_{HC}$']
    
    X_train_df=pd.DataFrame(data=X_train_rs,columns=input_variables_shap)
    
    explainer=shap.Explainer(model,X_train_df)  
    
    fs=20
    
    shap_values=explainer(X_train_df)
    
    def format_plot(shap_vals,title):
        fig,ax=plt.subplots(1)
        
        figure=shap.plots.beeswarm(shap_vals,show=None)
        
        figure.set_title(title,fontsize=fs)
        figure.tick_params(labelsize=fs-4)
        figure.set_xlabel('SHAP Value (impact on model output)',fontsize=fs-4)
        figure.set_xlim(-.35,.35)
        
    titles=['IMF $B_{X}$', 'IMF $B_{Y}$', 'IMF $B_{Z}$', 'IMF $|B|$']
    for i in range(4):
        format_plot(shap_values[:,:,i],titles[i])
            
def model_comparison_ca_2_var(trange,no_angle=False,two_plot=False,three_plot=False):
    
    ''' Create magnetic field plot with colored regions and alpha parameter to
    show how training/test set for data is built (Figure 3 in the manuscript)
    
    trange for manuscript: trange = ['2011-06-21 10:38:00', '2011-06-21 11:10:00']
    
    To make manuscript Figure 3: model_comparison_ca_2_var(trange,no_angle=True,two_plot=True)
    
    
    
    Can also make a time series of ANN predictions based on a time range (Figure 8)
    
    trange for manuscript plot: trange = ['2011-05-06 20:00:09','2011-05-06 22:00:00']
    
    To make manuscript Figure 8: model_comparison_ca_2_var(trange,no_angle=True,three_plot=True)
    
    '''
    
    
    fs=22
    lw=2.5
    ## How to find a trange to check out:
    
    #df_results=pd.read_pickle('df_attempt_10sec_40_ensemble_200_models.pkl')
    #trange=[convert_datetime_to_string(df_results.time.iloc[17980]-datetime.timedelta(minutes=120)),convert_datetime_to_string(df_results.time.iloc[17980]+datetime.timedelta(minutes=120))]
   
   
    def generate_model_dataframe(trange):
        
        tr=trange
        
        trange2=[convert_to_datetime(tr[0]),convert_to_datetime(tr[1])]
        
        df=pd.read_pickle(save_path+'full_data_w_boundaries_40.pkl')
        
        #filename_all='/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/df_attempt_ALL_MS_ensemble_2_models_norm.pkl'
        
        filename_all='/Users/bowersch/df_attempt_ALL_MS_ensemble_100_models_norm_30deg.pkl'
        
        #filename_all='/Users/bowersch/df_attempt_FULL_ensemble_50_models_norm_30deg.pkl'
        
        df_results=pd.read_pickle(filename_all)
        
        df=df[((df.time >= trange2[0]) & (df.time <= trange2[1]))]
        
        df_results=df_results[((df_results.time >= trange2[0]) & (df_results.time <= trange2[1]))]
        
        time_full=df.time
        
        bswx_ghost=np.zeros(len(time_full))+float('nan')
        bswy_ghost=np.zeros(len(time_full))+float('nan')
        bswz_ghost =np.zeros(len(time_full))+float('nan')
        magsw_ghost =np.zeros(len(time_full))+float('nan')
        bswx_err_ghost =np.zeros(len(time_full))+float('nan')
        bswy_err_ghost =np.zeros(len(time_full))+float('nan')
        bswz_err_ghost =np.zeros(len(time_full))+float('nan')
        magsw_err_ghost =np.zeros(len(time_full))+float('nan')

        
 
        for i in range(len(df_results)):
            
            gd=np.where(df_results.time.iloc[i]==time_full)[0]
            
            if np.size(gd)==1:
                bswx_ghost[gd]=df_results.bswx_pred.iloc[i]
                bswy_ghost[gd]=df_results.bswy_pred.iloc[i]
                bswz_ghost[gd]=df_results.bswz_pred.iloc[i]
                magsw_ghost[gd]=df_results.magsw_pred.iloc[i]
                bswx_err_ghost[gd]=df_results.bswx_err.iloc[i]
                bswy_err_ghost[gd]=df_results.bswy_err.iloc[i]
                bswz_err_ghost[gd]=df_results.bswz_err.iloc[i]
                magsw_err_ghost[gd]=df_results.magsw_err.iloc[i]
                
# =============================================================================
#                 bx_ghost[gd]=df_results.bswx.iloc[i]
#                 by_ghost[gd]=df_results.bswy.iloc[i]
#                 bz_ghost[gd]=df_results.bswz.iloc[i]
#                 mag_ghost[gd]=df_results.magsw.iloc[i]
# =============================================================================
                
                
        df[['bswx_pred','bswy_pred','bswz_pred','magsw_pred','bswx_err','bswy_err','bswz_err','magsw_err']]=np.transpose(np.vstack((bswx_ghost,bswy_ghost,bswz_ghost,magsw_ghost,bswx_err_ghost,bswy_err_ghost,bswz_err_ghost,magsw_err_ghost)))
        
        return df
    
    df=generate_model_dataframe(trange)
    ds=trange[0][0:10]
    from trying3 import plot_MESSENGER_trange,plot_MESSENGER_trange_3ax
    
    import matplotlib.patches as mpatches
    
    
    
    clock_angle_pred=np.arctan2(df.bswy_pred,df.bswz_pred)*180/np.pi
    
    cone_angle_pred=np.arccos(-df.bswx_pred/df.magsw_pred)*180/np.pi
    
    clock_angle=np.arctan2(df.magy,df.magz)*180/np.pi
    
    cone_angle=np.arccos(-df.magx/df.magamp)*180/np.pi
    
    
    # Find MS point to compare angle to:
        
    shift=df.shift(1)
    
    shift_n=df.shift(-1)
    
    t_n=np.where((shift_n.Type_num != 2) & (df.Type_num==2))[0]
    
    t=np.where((shift.Type_num != 2) & (df.Type_num==2))[0]
    
        
        
    if np.size(t_n)==1:
        t_n=t_n[0]
        if ((shift_n.Type_num.iloc[t_n]==4) | (shift_n.Type_num.iloc[t_n]==3)):
            index_ms=t_n
            clock_angle_ms=clock_angle.iloc[index_ms]
            cone_angle_ms=cone_angle.iloc[index_ms]
            

   
    if np.size(t)==1:
        t=t[0]
        if ((shift.Type_num.iloc[t]==4) | (shift.Type_num.iloc[t]==3)):
            index_ms=t
            
    
            
  
    bx_ms=df.magx.iloc[index_ms]
    by_ms=df.magy.iloc[index_ms]
    bz_ms=df.magz.iloc[index_ms]
    clock_angle_ms=clock_angle.iloc[index_ms]
    
    cone_angle_ms=cone_angle.iloc[index_ms]
    
    def angle_between_vectors(vector1, vector2):
        # Convert the input lists to arrays
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        # Dot the vectors
        dot_product = np.dot(vector1, vector2)

        # Calculate the magnitudes of the vectors
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Calculate the cosine of the angle between the vectors
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in radians
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    
    ang_diff=np.array([angle_between_vectors([bx_ms,by_ms,bz_ms], \
                                             [df.magx.iloc[a],df.magy.iloc[a],df.magz.iloc[a]])\
                       for a in range(len(df))])
    
            
    
    
    diff=df.shift(1)-df
    
    transitions=np.where(diff.Type_num != 0.0)[0]
    
    colors=['firebrick','royalblue','gold','orange','mediumpurple']
    
    if no_angle==False:
        fig,ax=plt.subplots(5,sharex=True)
        
    else:
        
        if three_plot==True:
                
            fig,ax=plt.subplots(3,sharex=True)
        
            ax[2].set_xlabel(ds,fontsize=fs)

    
    transitions=np.insert(transitions,0,0)
    
    transitions=np.insert(transitions,np.size(transitions)-1,np.size(df.time)-1)
    
    transitions=np.sort(transitions)
    filename='dataset_30_diff.pkl'
    a=pd.read_pickle(save_path+filename)
    
    time_tt=a.time.to_numpy()
    
    time_df=df.time.to_numpy()
    
    diff=np.abs(time_tt-time_df[0])
    
    min_diff=np.where(diff==np.min(diff))[0][0]
    
    bswx=a.bswx.iloc[min_diff]
    
    bswy=a.bswy.iloc[min_diff]
    
    bswz=a.bswz.iloc[min_diff]
    
    mag_sw1=a.magsw.iloc[min_diff]
    
    mag_sw2=np.sqrt(bswx**2+bswy**2+bswz**2)
    
    
    
    clock_angle_sw=np.arctan2(bswy,bswz)*180/np.pi
    
    cone_angle_sw=np.arccos(-bswx/mag_sw2)*180/np.pi
            
    ax[0].plot(df.time,df.magx,color='blue',alpha=.7,linewidth=lw,label='$B_{X}$')
    ax[0].plot(df.time,df.magy,color='green',alpha=.7,linewidth=lw,label='$B_{Y}$')
    ax[0].plot(df.time,df.magz,color='red',alpha=.7,linewidth=lw,label='$B_{Z}$')
    
    ax[0].axhline(y=0,color='black',linestyle='--',linewidth=.5)
    
    #ax[3].plot(df.time,ang_diff,color='black',linewidth=lw,label='\u03B1')
    #ax[3].set_ylim(0,180)
    
    #ax[3].axhline(y=30,color='red',linewidth=lw-.5,linestyle='--')
    
    #ax[3].axhline(y=75,color='gold',linewidth=lw-.5,linestyle='--')
    
    #ax[3].axhline(y=90,color='red',linewidth=lw-.5,linestyle='--')
    
    
    ylabels=["$B_{MSM'}$ (nT)",'Pred IMF (nT)','|B| (nT)','\u03B1 (deg)']#,'Clock Angle (deg)','Cone Angle (deg)']
    
    
    shade_in_transitions(transitions,ax[0],df)
    shade_in_transitions(transitions,ax[1],df)
    
    shade_in_transitions(transitions,ax[2],df)
    
    #shade_in_transitions(transitions,ax[3],df)
    
    
    
    if no_angle==False:
        
        shade_in_transitions(transitions,ax[3],df)
        shade_in_transitions(transitions,ax[4],df)
    
    
    ax[0].set_ylim(-60,60)
    ax[1].set_ylim(-60,60)
    ax[2].set_ylim(0,120)
    
    
    sigma_level=3
    
    

    
    
    ax[1].plot(df.time,df.bswx_pred-df.bswx_err*sigma_level,color='blue',alpha=.7,linewidth=lw-1,label='Pred IMF $B_{X}$')
    ax[1].plot(df.time,df.bswy_pred-df.bswy_err*sigma_level,color='green',alpha=.7,linewidth=lw-1,label='Pred IMF $B_{Y}$')
    ax[1].plot(df.time,df.bswz_pred-df.bswz_err*sigma_level,color='red',alpha=.7,linewidth=lw-1,label='Pred IMF $B_{Z}$')
    
    # Plot Uncertainty
    ax[1].plot(df.time,df.bswx_pred+df.bswx_err*sigma_level,color='blue',alpha=.7,linewidth=lw-1)
    ax[1].plot(df.time,df.bswy_pred+df.bswy_err*sigma_level,color='green',alpha=.7,linewidth=lw-1)
    ax[1].plot(df.time,df.bswz_pred+df.bswz_err*sigma_level,color='red',alpha=.7,linewidth=lw-1)
    
    ax[1].fill_between(df.time,df.bswx_pred+df.bswx_err*sigma_level,df.bswx_pred-df.bswx_err*sigma_level , interpolate=True, color='blue', alpha=0.5)
    ax[1].fill_between(df.time,df.bswy_pred+df.bswy_err*sigma_level,df.bswy_pred-df.bswy_err*sigma_level , interpolate=True, color='green', alpha=0.5)
    ax[1].fill_between(df.time,df.bswz_pred+df.bswz_err*sigma_level,df.bswz_pred-df.bswz_err*sigma_level , interpolate=True, color='red', alpha=0.5)
    
    

    

    
    
    ax[1].axhline(y=0,color='black',linestyle='--',linewidth=lw-1)
    
    #Plot Uncertainty
    ax[2].plot(df.time,df.magsw_pred-df.magsw_err*sigma_level,color='pink',label='Pred IMF |B|',linewidth=lw-1)
    ax[2].plot(df.time,df.magsw_pred+df.magsw_err*sigma_level,color='pink',linewidth=lw-1)
    ax[2].fill_between(df.time,df.magsw_pred-df.magsw_err*sigma_level,df.magsw_pred+df.magsw_err*sigma_level, interpolate=True,color='pink',alpha=0.8)
    
    
    
    ax[2].plot(df.time,df.magamp,color='black',label='Measured |B|',linewidth=lw)
    
    df_t=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/dataset_10_min_lim_40.pkl')
    df_t=df_t[((df_t.time > convert_to_datetime(trange[0])) & (df_t.time < convert_to_datetime(trange[1])))]
    diff=df_t.shift(1)-df_t
    
    #transitions=np.where(diff.bswx != 0.0)[0]
# Set if you want to plot the "actual" IMF conditions onto the plot as scatter points    
# =============================================================================
#     for i in transitions:
#         
#     
#         ax[1].scatter(df_t.time.iloc[i],df_t.bswx.iloc[i],color='blue')
#         ax[1].scatter(df_t.time.iloc[i],df_t.bswy.iloc[i],color='green')
#         ax[1].scatter(df_t.time.iloc[i],df_t.bswz.iloc[i],color='red')
#     
#         ax[2].scatter(df_t.time.iloc[i],df_t.magsw.iloc[i],color='pink')
# =============================================================================
        
    ax[0].set_xlim(convert_to_datetime(trange[0]),convert_to_datetime(trange[1]))
    
    
    
    if no_angle==False:
        ax[3].plot(df.time,clock_angle,color='black',label='Measured Clock Angle')
        
        ax[3].plot(df.time,clock_angle_pred,color='pink',label='Predicted IMF Clock Angle')
    
        
        
        ax[3].set_ylim(-180,180)
        
        ax[4].plot(df.time,cone_angle,color='black',label='Measured Cone Angle')
        
        ax[4].plot(df.time,cone_angle_pred,color='pink',label='Predicted IMF Cone Angle')
        
        ax[4].set_xlabel(ds,fontsize=fs)

        
        
        
        
        #ax[3].scatter(df.time.iloc[np.size(df.time)-1],clock_angle_sw,color='black')
    
        #ax[4].scatter(df.time.iloc[np.size(df.time)-1],cone_angle_sw,color='black')
        
    
            
        
        
        
        
        ax[4].set_ylim(-180,180)
        
        ax[4].set_ylim(0,180)
        
                
        shade_in_transitions(transitions,ax[3],df)
        
        shade_in_transitions(transitions,ax[4],df)
    

    for i in range(len(ax)):
        
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[i].tick_params(labelsize=fs-3)
        
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[i].tick_params(axis='both', which='major', length=8)
        ax[i].tick_params(axis='both', which='minor', length=4)
        
        
    
    
    for i in range(len(ax)):
        ax[i].set_ylabel(ylabels[i],fontsize=fs)
        ax[i].legend(loc='upper right',fontsize=fs)
        
        
    if two_plot==True:
        
        fig,ax=plt.subplots(2,sharex=True)
        
        ax[0].set_ylim(-60,60)
        ax[0].set_xlim(convert_to_datetime(trange[0]),convert_to_datetime(trange[1]))
        
        ax[0].plot(df.time,df.magx,color='blue',alpha=.7,linewidth=lw,label='$B_{X}$')
        ax[0].plot(df.time,df.magy,color='green',alpha=.7,linewidth=lw,label='$B_{Y}$')
        ax[0].plot(df.time,df.magz,color='red',alpha=.7,linewidth=lw,label='$B_{Z}$')
        
        ax[0].axhline(y=0,color='black',linestyle='--',linewidth=.5)
        
        ax[1].plot(df.time,ang_diff,color='black',linewidth=lw,label='\u03B1')
        ax[1].set_ylim(0,180)
        
        ax[1].axhline(y=30,color='red',linewidth=lw-.5,linestyle='--')
        
        shade_in_transitions(transitions,ax[0],df)
        shade_in_transitions(transitions,ax[1],df)
        ax[1].set_xlabel(ds,fontsize=fs)
        
        
        
        for i in range(len(ax)):
            
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax[i].tick_params(labelsize=fs-3)
            
            ax[i].xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            ax[i].yaxis.set_minor_locator(AutoMinorLocator())
            
            ax[i].tick_params(axis='both', which='major', length=8)
            ax[i].tick_params(axis='both', which='minor', length=4)
            
            
        ylabels=["$B_{MSM'}$ (nT)",'\u03B1 (deg)']
        
        for i in range(len(ax)):
            ax[i].set_ylabel(ylabels[i],fontsize=fs)
            ax[i].legend(loc='upper right',fontsize=fs)
            
            

        

    