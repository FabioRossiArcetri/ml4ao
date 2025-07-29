#!/usr/bin/python3

import os,sys
import numpy as np

from scipy import signal
from scipy import optimize
from scipy import special


def r0_wind_estimation(diameter_in, freq_in, psd_in,nmodes_in,outerScale_in):
    # ESTIMATE r0 and wind speed from psd
    # diameter_in is the telescope diameter. From, params.txt (PASATA) it is params['MAIN']['PIXEL_PITCH']*params['MAIN']['PIXEL_PUPIL']
    # freq_in is the "freq" output of compute_psd function
    # psd_in is the "psd" output from compute_psd function rescaled this way: psd*(2*np.pi/500.)**2.
    # nmodes_in is the number of corrected modes
    # outerScale is usually 20. (m) whithout better estimation
    nnMin=3
    turb_var,wind_speed = turb_var_and_wind_estimation(diameter_in, freq_in, psd_in, nmodes_in, add_power=False)
    r0 = r0_estimation_from_olvar(turb_var, nmodes_in, nnMin, diameter_in,outerScale_in)

    return r0,wind_speed

def compute_psd(ArrayIn,framerate):
    # Computes the PSD
    # If ArrayIn is 2D, it must be [nmodes,ntimes], else it must be a 1D array
    # ArrayIn is usually turbArray = comm + resMod (or POL) where comm is red from comm.fits and resMod is red from resMod.fits (Passata)
    # framerate is framerate=1.0/timestep
    if isinstance(ArrayIn,np.ndarray):
        if len(ArrayIn.shape) == 2 :
            nmodes=ArrayIn.shape[0]
            ntimes=ArrayIn.shape[1]
        elif len(ArrayIn.shape) == 1 :
            nmodes=0
            ntimes=len(ArrayIn)
        else:
            print('ERROR: compute_psd: ArrayIn must be 1D or 2D np.ndarray')
            sys.exit(1)
    else:
        print('ERROR: compute_psd: ArrayIn must be 1D or 2D np.ndarray')
        sys.exit(1)
    psd=np.zeros((int(ntimes/2),nmodes))

    for j in range(nmodes):
        freq,powerspectrum=signal.welch(ArrayIn[j,:],fs=framerate,window='hann',nfft=ntimes,scaling='density',nperseg=ntimes)

        freq=freq[1:]
        powerspectrum=powerspectrum[1:]
        psd[:,j]=powerspectrum

    return freq,psd


def turb_var_and_wind_estimation(diameter_in, freq_in, psd_in, nmodes_in, add_power=False):
    # Estimates the wind speed and the variance of the turbolence, by mode
    # Based on Guido's work
    # diameter_in: telescope diameter
    # freq_in: vector of the frequency sampling
    # psd_in: 2D array holding a PSD for each mode
    # nmodes_in: number of modes
    # add_power: when this is True tries to compute the share of power lost
    
    nPSD = psd_in.shape[0]
    fsample=freq_in[1]-freq_in[0]

    turb_var=np.zeros(nmodes_in)
    radial_degree=np.zeros(nmodes_in)
    f_cut=np.zeros(nmodes_in)
    power=np.zeros(nmodes_in)
    noise=np.zeros(nmodes_in)
    wind_speed=np.zeros(nmodes_in)
    missing_power=np.zeros(nmodes_in)
    for imode in range(nmodes_in):
        #TURBULENCE
        #Noise variance is a constant offset in the temporal PSD (temporally uncorrelated), mostly visible at high frequencies
        noise_level = np.mean(psd_in[round(nPSD/2.):,imode])
        TMPpsd=psd_in[:,imode]-noise_level
        TMPpsd[TMPpsd<0]=0
        turb_var[imode] = np.sum(TMPpsd)*fsample
        #WIND SPEED
        if imode > 2:
            #computes Zernike radial degree
            rd, af = zern_degree(imode+2)
            radial_degree[imode] = rd
            #function initial values and characteristic scale
            #cut-off frequency initial guess
            if imode > 10:
                v_init=np.mean(f_cut[f_cut > 0] * diameter_in / (0.3 * (rd+1.)))
                if v_init < 0 : 
                    v_init = 10.
            else:
                v_init=10.
            f_c_temp = 0.3 * (rd+1.) * v_init / diameter_in
            #initial guess
            P0 = [f_c_temp, turb_var[imode], noise_level*nPSD*fsample]
#            scale = [P0[0]/3., P0[1]/10., P0[2]/10.]

            solution=NelderMead_psd_estimation(freq_in, psd_in[:,imode],np.asarray(P0))
            f_cut[imode] = solution[0]
            power[imode] = solution[1]
            noise[imode] = solution[2]

            wind_speed[imode] = f_cut[imode] * diameter_in / (0.3 * (rd+1.))

    idx_pos=np.where(wind_speed > 0)[0]
    wind_speed_pos = wind_speed[idx_pos]
    wind_speed_mean = np.mean(wind_speed_pos)
    wind_speed_median = np.median(wind_speed_pos)

    #checks if some modes must be discarded:
    # 1) because the minimum frequency of the PSD is greater than its cut-off frequency.
    f_cut_mean = 0.3 * (radial_degree+1.) * wind_speed_mean / diameter_in
    idx_in_freq = np.where(f_cut_mean[idx_pos] >= min(freq_in))
    wind_speed_mean = np.mean(wind_speed_pos[idx_in_freq])
    wind_speed_median = np.median(wind_speed_pos[idx_in_freq])

    # 2) because is out of 3 sigma
    sigma_wind = np.std(wind_speed[idx_in_freq])
    wind_speed_pos_tmp=wind_speed_pos[idx_in_freq]
    idx_in_sigma1= np.where(wind_speed_pos_tmp <= wind_speed_mean+3.*sigma_wind)
    idx_in_sigma2= np.where(wind_speed_pos_tmp >= wind_speed_mean-3.*sigma_wind)
    idx_in_sigma=np.intersect1d(idx_in_sigma1,idx_in_sigma2)
    if len(idx_in_sigma) > 0 :
        wind_speed_mean = np.mean(wind_speed_pos_tmp[idx_in_sigma])
        wind_speed_median = np.median(wind_speed_pos_tmp[idx_in_sigma])
    else:
        wind_speed_mean=np.nan
        wind_speed_median=np.nan

    f_cut_mean = 0.3 * (radial_degree+1.) * wind_speed_mean / diameter_in
    for imode in range(nmodes_in) :
        idx_f_cut = closest(f_cut_mean[imode],freq_in)
        if freq_in[0] > 1/30. :
            missing_power[imode] = np.mean(psd_in[0:idx_f_cut+1,imode])*(freq_in[0]-1/30.)
    if add_power :
        turb_var += missing_power

    return turb_var,wind_speed_mean

def NelderMead_psd_estimation(freq_in, psd_in, init_variables):
    # Fitting of the PSD in input and a model of PSD defined by 3 linear segments, 
    # where low and hig temporal freqs are constant, medium freqs have f^(-17/3) slope
    # freq_in: vector of the frequency sampling
    # psd_in: imput PSD
    if isinstance(init_variables,np.ndarray):
        P0 = init_variables
#        P0 = init_variables[:,0]
#        scale = init_variables[*,1]
    elif np.isscalar(j) :
        P0 = [10., np.sum(psd), 25.]
#        scale = [5., np.sum(psd)/10., 5.]
    #max number of tries and tolerance
    nmax = 2000.0
    global psd_get
    psd_get=psd_in.copy()
    global freq_get
    freq_get=freq_in.copy()
    result = optimize.minimize(res_psd_estimation,P0,method='nelder-mead',options={"maxiter":len(P0)*nmax,"adaptive":True})
    solution = result['x']

    return solution

def closest(value_in,array_in,outboundflag=False):
    # closest returns the index within an array which is closest to the
    # user supplied value. If value is outside bounds of array, closest
    # returns -1.
    if outboundflag :
        nmax=np.max(array_in)
        if value_in > nmax:
            return -1
        nmin=np.min(array_in)
        if value_in < nmin:
            return -1
    index = (np.abs(array_in - value_in)).argmin()
    return index

def res_psd_estimation(variables):
    # defines the error metric to be minimized in the temporal PSD model fitting
    n_glob = len(psd_get)
    #checks on the variables
    #cut-off frequency > min(param.freq) > 0
    if variables[0] < min(freq_get) :
        variables[0] = min(freq_get)
    if variables[0] < 0 :
        variables[0] = min(freq_get)
    #power > 0
    if variables[1] < 0. :
        variables[1] = 1.
    if variables[2] < 0. :
        variables[2] = 0.1
    f_cut = variables[0]
    power = variables[1]
    noise = variables[2]

    #build the candidate PSD
    fsample = freq_get[1] - freq_get[0]
    est_psd = np.zeros(n_glob)
    idx_f_cut = closest(f_cut,freq_get)
    est_psd[idx_f_cut+1:] = freq_get[idx_f_cut+1:]**(-17./3.)
    est_psd[0:idx_f_cut+1] = est_psd[0:idx_f_cut+1]+freq_get[idx_f_cut]**(-17./3.)
    est_psd = est_psd * (power/fsample) / np.sum(est_psd)
    est_psd += noise/fsample/n_glob
    res = np.sum(np.abs(psd_get - est_psd))

    return res

def zern_degree(index):
    n=np.ceil(0.5 * (np.sqrt(8 * np.array(index) + 1) - 3)).astype(int)
    cn = n * (n + 1) / 2 + 1
    if n % 2 == 0:
        m = int(index - cn + 1) // 2 * 2
    else:
        m = int(index - cn) // 2 * 2 + 1
    radialDegree = n
    azimuthalFrequency = m
    return radialDegree, azimuthalFrequency

#def zern_degree(j):
#    n=int(0.5*(np.sqrt(8*j-7)-3))+1
#    cn = int(n*(n+1)/2)+1
#    if isinstance(n,np.ndarray):
#        idx_even = [idx for idx, item in np.ndenumerate(n) if item % 2 == 0]
#        idx_odd = [idx for idx, item in np.ndenumerate(n) if item % 2 != 0]
#        m = n*0
#        temp = j-cn
#        if len(idx_even) > 0 :
#            m[idx_even]=int((temp[idx_even]+1)/2)*2
#        if len(idx_odd) > 0 :
#            m[idx_odd]=int((temp[idx_odd])/2)*2+1
#    elif np.isscalar(j) :
#        if n % 2 == 0 :
#            m = int((j-cn+1)/2)*2
#        else:
#            m = int((j-cn)/2)*2+1
#    else:
#        print("ERROR in zern_degree: you must pass either a numpy array or a scalar")
#        sys.exit(1)
#    return n,m

def kolm_covar(j1, j2):
    # covariance between two zernike modes
    
    n1,m1=zern_degree(j1)
    n2,m2=zern_degree(j2)

    if (m1 != m2) or ((((j1+j2) % 2) != 0 ) and (m1 != 0)) :
        return 0
    #npn and nmn are both integer because m1=m2, so n1 and n2 have the
    #same parity.
    #Moreover, because n1,n2>=1 in this frame, npn>=0
    #and nmn>=0

    npn = int(n1 + n2)//2-1
    nmn = int(np.abs(n1 - n2))//2

    #the costant reported in Roddier 1990 is divided by 2*pi 
    #to convert rad in waves
    result = 0.057494899*(-1)**int((n1+n2-2*m1)//2)*np.sqrt((n1+1)*(n2+1))
    result = result*special.gamma(1/6)/(special.gamma(17/6))**2/special.gamma(29/6)
    c1=1
    c2=1
    if (npn > 0) :
        for i in range(npn):
            c1=c1/(29/6+i)*(1/6+i)
    if (nmn > 0) :
        for i in range(nmn):
            c2=c2/(17/6+i)*(-11/6+i)
    return result*c1*c2*(-1)**nmn

def kolm_mcovar(max_index) :
    # computes a covariance matrix of the zernike modes
    n_elem = max_index-1
    if n_elem < 1:
        return 0
    result = np.zeros((n_elem,n_elem))
    for i in range(n_elem):
        for j in range(i,n_elem):
            result[j,i] = kolm_covar(j+2,i+2)
            result[i,j] = result[j,i]
    return result

def diag_matrix(M):
    if not isinstance(M,np.ndarray):
        print("ERROR: diag_matrix: you must pass a numpy array")
        sys.exit(1)
    size1=M.shape[0]
    size2=M.shape[1]
    if size1 != size2 :
        print("ERROR: diag_matrix: matrix is not squared")
        sys.exit(1)
    diagonal=np.zeros(size1)
    for i in range(size1):
        diagonal[i]=M[i,i]
    return diagonal

def diagonalize_matrix(M):
    # Find Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Check Diagonalizability
    if not np.all(np.iscomplex(eigenvalues)):
        #Create Diagonal Matrix
        D = np.diag(eigenvalues)
        #Find the Inverse of Eigenvector Matrix
        P_inv = np.linalg.inv(eigenvectors)
        #Diagonalize the Matrix
        diagonalized_matrix = np.dot(P_inv, np.dot(A, eigenvectors))
    else:
        print("Matrix is not diagonalizable.")
        sys.exit(1)
    return diagonalized_matrix

def r0_est_fit_errfunc(r0i):
    # error metric to be optimized when fitting the modal variances of the atmosphere
    r0 = np.abs(r0i)
    theoVar = (4.*np.pi**2) * (diam_get/r0)**(5./3.) * theovar1
    ResError = np.sum(np.abs(olvarMean - theoVar)/olvarMean)
    return ResError

def r0_estimation_from_olvar(olvar, nmodes, nnMin, DpupM, outerScale):
    # Based on Fernando Quiros Pacheco work
    # estimates r0 from the POL modal variances
    modes_idx=np.arange(nmodes)
    nnMax=int((np.sqrt(8*nmodes-7)-1))//2
    nnRange=[nnMin,nnMax-1]
    if (nnRange[1]-nnRange[0]+1) <= 0 :
        print('Not enough OLmodes to perform r0 estimation')
        return 0

    Zern_number = np.arange(nmodes)+2
    tmp=np.sqrt(8*Zern_number-7)-1
    tmp=tmp//2
    nn=tmp.astype(int)
    nnValues = np.arange(nnRange[1]-nnRange[0]+1)+nnRange[0]
    norders  = len(nnValues)

    #Experimental data: the average per radial order of the coeff variances is computed:
    global olvarMean
    olvarMean = np.zeros(norders)
    nn_count  = np.zeros(norders)
    for j in range(norders):
        for i in range(nmodes):
            if nn[i] == nnValues[j]:
                olvarMean[j] = olvarMean[j] + olvar[i]
                nn_count[j]  = nn_count[j] + 1
        olvarMean[j] = olvarMean[j] / float(nn_count[j])

    #Theoretical data: a single variance per radial order:
    #Fist zernike of each radial order.
    FirstZerns=np.zeros(len(nnValues)).astype('int')
    for iterme in range(len(nnValues)):
        FirstZerns[iterme] = int(nnValues[iterme]*(nnValues[iterme]+1))//2+1

    #kolmogorov or von karman spectrum
    #outerScale is a scalar, else transalte function VON_COVAR
    global theovar1
    theovar1=np.zeros(len(FirstZerns))
    K=diag_matrix(kolm_mcovar(nmodes+1))
    tmp=FirstZerns-2 
    print(tmp)
    for iterme in range(len(FirstZerns)):
        theovar1[iterme] = K[tmp[iterme]]

#    if np.isscalar(outerScale):
#        theovar1 = diag_matrix(kolm_mcovar(nmodes+1))[FirstZerns-2]
#    else:
#        theovar1 = np.zeros(len(FirstZerns))
#        L0norm = (outerScale/DpupM)
#        for i in range(FirstZerns):
#            theovar1[i] = VON_COVAR(FirstZerns[i],FirstZerns[i],L0norm,/double)

    x = np.arange(len(theovar1))+1
    tot_theovar1=np.sum(theovar1)
    if (len(theovar1) > 6) and (len(theovar1) <= 16) :
        theovar1 *= x**(-1/16.)
        theovar1[0] += (tot_theovar1-np.sum(theovar1))
    if (len(theovar1) > 16) :
        theovar1 *= x**(-1/14.)
        theovar1[0] += (tot_theovar1-np.sum(theovar1))
    #Find best fit
    global diam_get
    diam_get = DpupM

    bnds=optimize.Bounds(0.01, 1.0)
    result = optimize.minimize(r0_est_fit_errfunc,(0.2),bounds=bnds,method='nelder-mead',options={"maxiter":500})
    print(result)
    r0fit = result['x']

    return r0fit

def sr_from_slopes(deltaComm, framerate, seeing, lambda_c=1650.0, TILT_FREE=False, VERBOSE=False):
    # Marechal estimation of sr from WFS error
    # lambda: Wvelength in nm. Default = 1650 nm.
    lambda_c=1650
    freq_max=framerate/2.
    nmodes=deltaComm.shape[0]
    ntimes=deltaComm.shape[1]
    nm2torad2 = (2.0*np.pi/lambda_c)**2.0

    #Varianza temporale
    clvar0=np.std(deltaComm,axis=1)

    freq_resm,psd_resm = compute_psd(deltaComm,framerate)
    psd_nelem=psd_resm.shape[0]

    noise_level = np.zeros(nmodes)
    if freq_max > 250:
        th=int(np.round(psd_nelem/2))
        tmp=np.mean(psd_resm[th:,:],axis=0)
    else:
        th=int(np.round(3*psd_nelem/4))
        tmp=np.mean(psd_resm[th:,:],axis=0)
    noise_level= tmp*framerate/2
    clvar=clvar0-noise_level
    clvar[clvar < 0.] = 0.

    if (np.sum(clvar) < 10) or (np.sum(noise_level)<10):
        print('WARNING: sr_from_slopes on may be in error.')

    if TILT_FREE :
    #remove tip and tilt if the tilt_free keyword is set.
        clvar = clvar[2:]

    rad2asec = 3600.0*180.0/np.pi
    asec2rad = 1.0/rad2asec

    seeing_rad = seeing*asec2rad
    r0500 = 0.976*0.0000005/seeing_rad # Fried's r0 @ 500 nm
    r0LAM = r0500*(lambda_c/500.0)**(6.0/5.0)
    fitting_error = 0.2313*(8.222/np.sqrt(nmodes*4.0/np.pi)/r0LAM)**(5.0/3.0)

    SR=np.exp(-1.0*(np.sum(clvar)*nm2torad2+fitting_error))

    if VERBOSE:
        print('SR@'+str(lambda_c)+'nm:'+str(SR))
        print('fitting error [nm]: '+str(np.sqrt(fitting_error/nm2torad2)))
        print('residual on corrected modes [nm]: '+str(np.sqrt(np.sum(clvar))))
        if TILT_FREE:
            print('residual on tip&tilt [nm]: '+str(np.sqrt(np.sum(clvar[0:2]))))
            print('SR@'+str(lambda_c)+'nm w/o tip&tilt: '+str(np.exp(-1.0*(np.sum(clvar[2:])*nm2torad2+fitting_error))))
        print('noise level in slopes [nm]: '+str(np.sqrt(np.sum(noise_level))))

    return SR
