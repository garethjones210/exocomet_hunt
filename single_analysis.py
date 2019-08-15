#!/usr/bin/env python3
import os; os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from functools import reduce
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser(description='Analyse target lightcurve.')
parser.add_argument(help='Target lightcurve file',nargs=1,dest='fits_file')
parser.add_argument('-n', help='No graphical output', action='store_true')
parser.add_argument('-q', help='Keep only points with SAP_QUALITY=0',action='store_true')
parser.add_argument('-r', help='Number of rounds of filtering', default=2, type=int)
parser.add_argument('-tm', help='Remove times which contain too many bad transits', action='store_true')
parser.add_argument('-ll', help='Lower time limit of bad transit removal', default=None, type=int)
parser.add_argument('-ul', help='Upper time limit of bad transit removal', default=None, type=int)
parser.add_argument('-s', help='Calculate T values in steps rather than the complete array', action='store_true')

args = parser.parse_args()

# Print if time arguments were not included
if args.tm:
    if args.ll == None or args.ul == None:
        print("Upper and/or lower limits for time removal required using -ul and -ll with integer respectively")
        sys.exit()

dip = 0

for g in range(args.r):
    run_time = time.time()
    
    print("This is run:", g+1)
    try:
        if args.fits_file[0].endswith('.fits'):
            # Assumes flux values are imported
            table = import_lightcurve(args.fits_file[0], args.q)
            t, flux, quality, real = clean_data(table)
        else:
            # Assumes magnitude values are imported instead of flux
            table = import_fullframeimage(args.fits_file[0])
            t, mag, _, real = clean_data(table)
            # Convert TESS magnitudes to flux
            flux =  10**(-0.4 * (mag - 20.60654144))
            quality = None
    except:
        print("File not loaded")
        sys.exit()

    timestep = calculate_timestep(table)
    N = len(t)
    ones = np.ones(N)

    flux = normalise_flux(flux)

    filteredflux = fourier_filter(flux, 8)
    A_mag = np.abs(np.fft.rfft(flux))
    periodicnoise = flux - filteredflux
    
    # Remove dip when filtering
    real_tmp = np.copy(real)
    real_tmp[dip] = 0
    
    flux_ls = np.copy(flux)
    lombscargle_filter(t, flux_ls, real_tmp, 0.05)
    periodicnoise_ls = flux - flux_ls
    flux_ls = flux_ls * real
    
    # Number of data points needed, relative to a 30 minute timestep, to make a width of 1.25 days
    step = 1
    multiple = 1/(48*timestep)
    if args.s:
        step = int(round(multiple))
    data_points = int(round(60*multiple))

#     T1 = test_statistic_array(filteredflux, data_point, step)
    T = test_statistic_array(flux_ls, data_points, step)
    
    if args.tm:
        time_removal(t, T, args.ll, args.ul)

    # Find minimum test statistic value, and its location
    m, n = np.unravel_index(T.argmin(), T.shape)
    minT = T[m, n]
    minT_time = t[n]
    minT_duration = m * timestep
    
    # Remove dip for calculation of noise within data
    T_SN = np.copy(T)
    half_m = int(round(0.5 * m))
    
    try:
        T_SN[:,n-half_m : n+half_m] = 0
    except:
        print("Minimum too close to edge for data exclusion in sigma calculation")
        pass
    
    # Noise calculation
    data = nonzero(T_SN[m])
    
    print("Maximum transit chance:")
    print("   Time =",round(minT_time,2),"days.")
    print("   Duration =",round(minT_duration,2),"days.")
    print("   T =",round(minT,1))
    print("   T/sigma =",round(minT/data.std(),1))

    trans_start = n - math.floor((m-1)/2)
    trans_end = trans_start + m
    print("Transit depth =",round(flux[trans_start:trans_end].mean(),6))
    
    dip = slice(n-m, n+m)
        
#     print("Run took:", time.time()-run_time)
    print("")


# Transit shape calculation
try:
    scores, width1, width2, x2, t2, y2, w2 = calc_shape(m, n, t, flux_ls, cutout_half_width=4, extra_info=True)

    print(scores)
    asym = scores[0]/scores[1]
    print("Asym score:", round(asym, 4))
    
    try:
        q2 = quality[n-4*m:n+4*m]
        qual_flags = reduce(lambda a,b: a|b, q2)
        print("Quality flags:", qual_flags)
    except:
        pass

    print("")
    print("Width1 =", width1,"and width2 =", width2)
    print("")
except:
    asym=-6
    width1=-6
    width2=-6
    pass
    
# Classify events
try:
    print(classify(m, n, real, asym, width1, width2))
except:
    print("Could not classify")
    pass
    
# Skip plotting if no graphical output set
if args.n:
    sys.exit()


# #plt.xkcd()
# fig1,axarr = plt.subplots(4)
# axarr[0].plot(A_mag)
# axarr[1].plot(t,flux+ones,t,periodicnoise_ls+ones)
# axarr[2].plot(t,flux_ls+ones)
# cax = axarr[3].imshow(T)
# axarr[3].set_aspect('auto')
# fig1.colorbar(cax)

# fig1.subplots_adjust(hspace=0.3)

# #params = double_gaussian_curve_fit(T)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# #T_test_nonzero = np.array(data)
# #_,bins,_ = ax2.hist(T_test_nonzero,bins=100,log=True)
# #y = np.maximum(bimodal(bins,*params),10)
# #ax2.plot(bins,y)
# try:
#     ax2.plot(t2,x2,t2,y2,t2,w2)
# except:
#     pass

# Plots
fig = plt.figure(figsize = (14, 8))

gs = gridspec.GridSpec(4, 2, wspace=0.2, hspace=0.2)

# Frequency spectrum
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(A_mag)

# Raw data and noise
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(t,flux+ones,t,periodicnoise_ls+ones)

# Filtered data
ax3 = fig.add_subplot(gs[2,0])
ax3.plot(t,flux_ls+ones)

# T chart
ax4 = fig.add_subplot(gs[3,0])
cax = ax4.imshow(T)
ax4.set_aspect('auto')
fig.colorbar(cax)

# Plot of dip
ax5 = fig.add_subplot(gs[:,1])
try:
    ax5.plot(t2,x2,t2,y2,t2,w2)
except:
    pass

# print("Time taken:", time.time()-start_time)
plt.show()

