#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons
import os
os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
import multiprocessing
import sys
import traceback
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser(description='Analyse lightcurves in target directory.')
parser.add_argument(help='Target directory(s)',
                        default='.',nargs='+',dest='path')

parser.add_argument('-t', help='Number of threads to use',default=1,
                        dest='threads',type=int)

parser.add_argument('-o',default='output.txt',dest='of',help='Output file')

parser.add_argument('-q', help='Keep only points with SAP_QUALITY=0',action='store_true')
parser.add_argument('-s', help='Calculate T values in steps rather than the complete array', action='store_true')
parser.add_argument('-r', help='Number of rounds of filtering', default=2, type=int)

parser.add_argument('-tm', help='Remove times which contain too many bad transits', action='store_true')
parser.add_argument('-ll', help='Lower time limit of bad transit removal', default=None, type=int)
parser.add_argument('-ul', help='Upper time limit of bad transit removal', default=None, type=int)

# Get directories from command line arguments
args = parser.parse_args()

# Print if time arguments were not included
if args.tm:
    if args.ll == None or args.ul == None:
        print("Upper and/or lower limits for time removal required using -ul and -ll with integer respectively")
        sys.exit()

paths = []
for path in args.path:
    paths.append( os.path.expanduser(path) )

# Prepare multithreading
m = multiprocessing.Manager()
lock = m.Lock()

def process_file(f_path):
    dip = 0
    try:
        f = os.path.basename(f_path)
        
        if f_path.endswith('.fits'):
            # Assumes flux values are imported
            table = import_lightcurve(f_path, args.q)
            mag = 0
        else:
            # Assumes magnitude values are imported instead of flux
            table = import_fullframeimage(f_path)
            mag = 1

        if len(table) > 120:
            for _ in range (args.r):
                t, flux, _, real = clean_data(table)
                if mag == 1:
                    # Convert TESS magnitudes to flux
                    DIA_flux =  10**(-0.4*(flux - 20.60654144))
                    flux = normalise_flux(DIA_flux)
                else:
                    flux = normalise_flux(flux)
                
                timestep = calculate_timestep(table)
                
                # Remove dip when filtering
                real_tmp = np.copy(real)
                real_tmp[dip] = 0
                
                # Number of data points needed, relative to a 30 minute timestep, to make a width of 1.25 days
                step = 1
                multiple = 1/(48 * timestep)
                if args.s:
                    step = int(round(multiple))
                data_points = int(round(60 * multiple))
                
                lombscargle_filter(t, flux, real_tmp, 0.05)
                flux = flux * real
                T = test_statistic_array(flux, data_points, step)
                
                if args.tm:
                    time_removal(t, T, args.ll, args.ul)
                
                # Determine width and time of dip respectively
                m, n = np.unravel_index(T.argmin(), T.shape)
                
                # Remove dip for calculation of noise within data
                Tn = np.copy(T)
                half_m = int(round(0.5 * m))
                
                try:
                    Tn[:,n-half_m : n+half_m] = 0
                except:
                    pass
                
                # Noise calculation
                Ts = nonzero(Tn[m]).std()
                
                dip = slice(n-m, n+m)
                Tm = T[m, n]
                
                # Break from loop if signal is weak
                if Tm/Ts > -4:
                    break
                
            Tm_time = t[n]
            Tm_duration = m * timestep
            Tm_start = n - math.floor((m - 1)/2)
            Tm_end = Tm_start + m
            Tm_depth = flux[Tm_start:Tm_end].mean()

            asym, width1, width2 = calc_shape(m, n, t, flux, cutout_half_width=4)
            s = classify(m, n, real, asym, width1, width2)

            result_str =\
                    f+' '+\
                    ' '.join([str(round(a,8)) for a in
                        [Tm, Tm/Ts, Tm_time,
                        asym,width1,width2,
                        Tm_duration,Tm_depth]])+\
                    ' '+s
        else:
            result_str = f+' 0 0 0 0 0 0 0 0 notEnoughData'

        lock.acquire()
        with open(args.of,'a') as out_file:
            out_file.write(result_str+'\n')
        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting",file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file "+f_path,file=sys.stderr)
        traceback.print_exc()



pool = multiprocessing.Pool(processes=args.threads)

for path in paths:
    if not os.path.isdir(path):
        print(path,'not a directory, skipping.',file=sys.stderr)
        continue

    fits_files = [f for f in os.listdir(path) if f.endswith('.fits') or f.endswith('.lc')]
    file_paths = [os.path.join(path,f) for f in fits_files]

    pool.map(process_file, file_paths)

# print("Time taken is:", time.time()-start_time)
    
