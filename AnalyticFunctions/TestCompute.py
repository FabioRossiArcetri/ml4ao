#!/usr/bin/python3

import os,sys,glob
import re, shutil, tempfile
import json
from collections.abc import Iterable
import math
import datetime as dt
import numpy as np
import pandas as pd
from astropy.io import fits
from AO_r0libs import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outfilename='RUN_V2_ANALYTIC_ESTIMATES.csv'

samples_to_remove=101

outerScale = 20.

DATAPATH='/home/turchi/TESTME/V2/'
DATAPATH='/luthien-raid/guido/results4mlV2/'

fitslist=['params.txt','deltaComm.fits', 'resVar.fits', 'srRes.fits', 'comm.fits', 'meanFlux.fits', 'resMod.fits']

def sed_inplace(filename, pattern, repl):
    pattern_compiled = re.compile(pattern)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename+'.json')

#####################################################################

DIRLIST=os.listdir(DATAPATH)
#Remove non directories. I assume all directories are passata ones.
for itemdir in DIRLIST.copy():
    if not os.path.isdir(os.path.join(DATAPATH,itemdir)):
        DIRLIST.remove(itemdir)
DIRLIST.sort()

OutputDf=pd.DataFrame(data=[],index=[])

FIRSTRUN=True
for itemdir in DIRLIST:
    DATAFULLPATH=os.path.join(DATAPATH,itemdir)
    print(DATAFULLPATH)
    print('Reading '+'params.txt'+' in '+DATAFULLPATH)
    FILEFULLPATH=os.path.join(DATAFULLPATH,'params.txt')
    datestr=os.path.basename(os.path.normpath(DATAFULLPATH))
    basedate=datestr.split('.')[0]
    dtbasedate=dt.datetime.strptime(basedate, '%Y%m%d_%H%M%S')
    ts=int(dtbasedate.timestamp())
    print(dtbasedate)

    #LOAD PARAMS FILE
    paramsfileraw=os.path.join(DATAFULLPATH,'params.txt')
    paramsfilejson=paramsfileraw+'.json'
    if os.path.isfile(paramsfilejson):
        os.remove(paramsfilejson)
    if not os.path.isfile(paramsfilejson):
        sed_inplace(paramsfileraw, r'Inf', '-99999')
    f=open(paramsfilejson)
    try:
        params=json.load(f)
    except:
        with open("failed_json_tracking.log", "a") as lgf:
            lgf.write(DATAFULLPATH+'\n')
        print('ERROR READING params.txt.json')
        f.close()
        sys.exit(0)
    f.close()
    for mainkey in params.keys():
        for subkey in params[mainkey].keys():
            if isinstance(params[mainkey][subkey], Iterable):
                sub_list = [math.inf if item == -99999 else item for item in params[mainkey][subkey]]
                params[mainkey][subkey]=sub_list
            else:
                if params[mainkey][subkey] == -99999 :
                    params[mainkey][subkey]=math.inf

    Seeing=params['SEEING']['CONSTANT']
    Cn2_list=[params['ATMO']['CN2'][0],params['ATMO']['CN2'][1],params['ATMO']['CN2'][2],params['ATMO']['CN2'][3]]
    WS_list=params['WIND_SPEED']['CONSTANT']
    R_mag=params['WFS_SOURCE']['MAGNITUDE']
    timestep=params['MAIN']['TIME_STEP']
    framerate=1.0/timestep
    diameter=params['MAIN']['PIXEL_PITCH']*params['MAIN']['PIXEL_PUPIL']
    total_time=params['MAIN']['TOTAL_TIME']
    controlmodes=params['MODALREC']['NMODES']

    FILEFULLPATH=os.path.join(DATAFULLPATH,'srRes.fits')
    hdul = fits.open(FILEFULLPATH,ignore_missing_simple=True)
    tmp=hdul[0].data.copy()
    #fits files are big endian
    srtrue=tmp.astype(np.float32)
    srtrue=srtrue[samples_to_remove:]
    srtrue=np.mean(srtrue)
    hdul.close()
    del hdul

    FILEFULLPATH=os.path.join(DATAFULLPATH,'comm.fits')
    hdul = fits.open(FILEFULLPATH,ignore_missing_simple=True)
    tmp=hdul[0].data.copy()
    #fits files are big endian
    comm=tmp.astype(np.float32)
    comm=comm[:,samples_to_remove:]
    hdul.close()
    del hdul

    FILEFULLPATH=os.path.join(DATAFULLPATH,'deltaComm.fits')
    hdul = fits.open(FILEFULLPATH,ignore_missing_simple=True)
    tmp=hdul[0].data.copy()
    #fits files are big endian
    deltaComm=tmp.astype(np.float32)
    deltaComm=deltaComm[:,samples_to_remove:]
    hdul.close()
    del hdul

    FILEFULLPATH=os.path.join(DATAFULLPATH,'resMod.fits')
    hdul = fits.open(FILEFULLPATH,ignore_missing_simple=True)
    tmp=hdul[0].data.copy()
    #fits files are big endian
    resMod=tmp.astype(np.float32)
    resMod=resMod[:,samples_to_remove:]
    hdul.close()
    del hdul

#    print(comm.shape)
#    print(deltaComm.shape)
#    print(resMod.shape)
    nmodes=comm.shape[0]
    ntimes=comm.shape[1]
    #resMod seems 630:1801
    resMod=resMod[0:nmodes,:]

    olArray = comm + deltaComm
    turbArray = comm + resMod

    freq,psd = compute_psd(turbArray,framerate)

    r0_est, wind_speed_est=r0_wind_estimation(diameter, freq, psd*(2*np.pi/500.)**2.,nmodes,outerScale)

    seeing_est =  0.9759 * 0.5/(r0_est*4.848)
    
    r0_true = 0.9759 * 0.5/(Seeing*4.848)

    #seeToko = sqrt(1- 2.183 * (r0/L0)^0.356)*seeing

    SR=sr_from_slopes(deltaComm,framerate,seeing_est,lambda_c=1650)
#    SR=sr_from_slopes(deltaComm,framerate,Seeing,lambda_c=1650)

    print('wind speed (TRUE):  ', WS_list)
    print('wind speed (EST):   ', wind_speed_est)
    print('r0 TRUE[m]:         ', r0_true)
    print('r0 EST[m]:          ', r0_est)
    print('seeing TRUE[arcsec]:', Seeing)
    print('seeing EST[arcsec]: ', seeing_est)
    print('SR TRUE[frac]:      ', srtrue)
    print('SR EST[frac]:       ', SR)
    print('total time [s]:     ', total_time)
    print('controlled modes    ', controlmodes)

    tmpOutDf=pd.DataFrame(index=[dtbasedate],data=[[np.mean(WS_list),wind_speed_est,r0_true,r0_est[0],Seeing,seeing_est[0],srtrue,SR[0],total_time,controlmodes]],columns=['wind speed (TRUE)','wind speed (EST)','r0 TRUE[m]','r0 EST[m]','seeing TRUE[arcsec]','seeing EST[arcsec]','SR TRUE[frac]','SR EST[frac]','total time [s]','controlled modes'])
    tmpOutDf.index.name='DateTime'

#    OutputDf=pd.concat([OutputDf,tmpOutDf])

    if FIRSTRUN:
        tmpOutDf.to_csv(outfilename,header=True,index=True)
        FIRSTRUN=False
    else:
        tmpOutDf.to_csv(outfilename,header=False,index=True,mode='a')
        FIRSTRUN=False

    


