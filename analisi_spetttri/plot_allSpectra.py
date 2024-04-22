import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os.path as ospath
from astropy.time import Time

import sys
sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
import matplotlib as mpl

mpl.rcParams['font.size']=15  #!!!!!!!!!!!!!!!!!!!!!!!!!!

counts_array, dead_array, start_array, livetime_array, norm_array, peak_array, sigma_array = [], [], [], [], [], [], []

def  plotAllSpectra(InputFileName):

    f=open(InputFileName)
    p=histogramSimo()
    n_spectra=0
    base_path=''
    legend=''
    show_legend=0
    #calP0=-0.0013498026638486778  #calP0Err= 3.3894706711692284e-05
    #calP1=0.0032116875215051385   #calP1Err= 3.284553141476064e-08
    P0_ = None
    P1_ = None

    compute_rate=0 
    time=1
    normalize=0
    low=0.
    up=0.

    fig=plt.figure(1, (10,10))
    ax = fig.subplots()
    for line in f:

        
        line=line.strip('\n')
        line=" ".join(line.split()) # rimuove tutti gli spazi doppi????? 
       # print(line)
        if len(line)==0:
            continue
        if line[0]=='#':
            continue
        splitted=line.split('=')
        print("splitted[0]=",splitted[0])
        if splitted[0]=='BASE_PATH':
           base_path=splitted[1]
           print("BASE_PATH=",base_path)
        
        if splitted[0]=='FILE':

            file_split=splitted[1].split(' ')
            print('file_split',file_split)
            filename=ospath.join(base_path,file_split[0])
            fileFormat=file_split[1]
            print("reading file",filename)
            print ("file type = ",fileFormat)
            legend=filename.split('/')[-1]
            n_spectra+=1
            
            p.read_from_file(filename, fileFormat )

        if   splitted[0]=='LEGEND':   
              legend=splitted[1] 
              print ("legend=",legend)
              show_legend=1
        if splitted[0]=="P0":
            P0_=float(splitted[1])
        if splitted[0]=="P1":
            P1_=float(splitted[1])
            print('taking P1 !!!!!!!!!!!',P1_)

                    
        if splitted[0]=="ACQ_TIME":
            compute_time=1
            if fileFormat=='sdd':
                time=p.sdd_liveTime
            else:    
                time=float(splitted[1])
       

        if  splitted[0]=="NORM_PEAK":
            #normalize peak height to 1
            print("!!!")
            normalize=1
            peak_limits=splitted[1].split(' ')
            if len(peak_limits)!=2:
                print("NORM_PEAK wrong limits... exit!")
                exit()
            low=float(peak_limits[0])
            up=float(peak_limits[1])
            if (up<=low):
                print("NORM_PEAK wrong limits... exit!")
                exit()
            print ('up=',up, "low = ",low)
            print("NORMALIZZO!!!!!")
            p.normalize(low,up)
            
            

                
 
        if splitted[0]=="ADD_PLOT": 
            # plot istogramma?
            if fileFormat=='sdd':
                calP0=-0.03544731540487446
                calP1=0.0015013787118821926
            if fileFormat=='npz':
                calP1=0.0032132721459619882
                calP0=-0.003201340833319255
            if P0_ != None:
                calP0 = P0_
            if P1_ != None:
                calP1 = P1_
            print("calP1=",calP1,"  calP0=",calP0)    
            p.bins=p.bins*calP1+calP0

            if compute_rate==1:
                 p.counts=p.counts/time
                 compute_rate=0
                 time=1
                 plt.ylabel('events/s')
            if normalize==1:
                 #print("NORMALIZZO!!!!!")
                 #p.normalize(low,up)
                 normalize=0
                 low=0.
                 up=0.
                 plt.ylabel('normalized rate')
            if show_legend==1:
                p.plot(ax,legend)
            else:
                p.plot(ax,None)
            plt.xlabel('energy [keV]')
            
            plt.ylabel('norm. counts') # non so perche', ,ma nell'if non funziona!
            plt.xlim(0.5,6)
            plt.ylim(7e-3,1.1)
            plt.yscale('log')
            plt.legend()


        
        if splitted[0]=="FIT":    
            fit_parameter=splitted[1].split(' ')
            
            min_x = float(fit_parameter[0])
            max_x = float(fit_parameter[1])
            amplitude = float(fit_parameter[2])
            peak = float(fit_parameter[3])
            sigma = float(fit_parameter[4])
            par, cov, chi2 = fitSimo.fit_Gaushistogram(p.counts, p.bins, xmin=min_x,xmax=max_x, initial_pars=[amplitude,peak,sigma], parsBoundsLow=-np.inf, parsBoundsUp=np.inf )
            x=np.linspace(min_x,max_x,1000)
            if show_legend==1:
                label='peak = '+"%.3f"%par[1]+' keV'+'\n'+'sigma = '+"%.3f"%par[2]+' keV'
            else:
                label=None
            ax.plot(x,fitSimo.gaussian_model(x,par[0],par[1],par[2]),label=label)
            print(' ')
            print('FIT PARAMETERS')
            print('Gaussian norm = ', "%.5f"%par[0],' +- ',"%.5f"%np.sqrt(cov[0][0]),' keV')
            print('Gaussian peak = ', "%.5f"%par[1],' +- ',"%.5f"%np.sqrt(cov[1][1]),' keV')
            print('Gaussian sigma = ', "%.5f"%par[2],' +- ',"%.5f"%np.sqrt(cov[2][2]),' keV')
            print(' ')
            print('         ####################################################################################')
            print('         ####################################################################################')

            # plt.xlabel('energy [keV]')
            
            # plt.ylabel('events/s') # non so perche', ,ma nell'if non funziona!

            plt.legend()

            counts_array.append(p.sdd_fastCounts)
            dead_array.append(p.sdd_deadTime)
            start_array.append(p.sdd_start)
            livetime_array.append(p.sdd_liveTime)
            norm_array.append(par[0])
            peak_array.append(par[1])
            sigma_array.append(par[2])


        if splitted[0]=='STABILITY':
            time_array = []
            for s in start_array:
                time_obj = Time(s, format='mjd', scale='utc')
                time_converted = time_obj.iso.split('T')[0]
                time_array.append(time_converted.split('.')[0])
            rate=[]
            for i in range(len(counts_array)):
                rate.append(counts_array[i]/livetime_array[i])
            rate_mean = np.mean(rate)
            rate_rms = np.std(rate, ddof=1)
            norm_mean = np.mean(norm_array)
            norm_rms = np.std(norm_array, ddof=1)
            peak_mean = np.mean(peak_array)
            peak_rms = np.std(peak_array, ddof=1)
            fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(12,8))
            #fig.suptitle('STABILITY')
            ax1.plot(start_array, (rate-rate_mean)/rate_mean*100.,marker='o',color='blue')
            ax1.set_ylabel('RATE' +'\n'+'res % [Hz]')
            ax1.axhline(y=0,color='black',label="%.3f"%rate_mean+' Hz'+'\n'+'rms= '+"%.3f"%rate_rms,linewidth=3)
            ax1.grid()
            ax1.legend()
            ax2.plot(start_array, (norm_array-norm_mean)/norm_mean*100.,marker='o',color='green')
            ax2.set_ylabel('NORM' +'\n'+'res % [counts]')
            ax2.axhline(y=0,color='black',label="%.3f"%norm_mean+'\n'+'rms= '+"%.3f"%norm_rms,linewidth=3)
            ax2.grid()
            ax2.legend()
            ax3.plot(start_array, (peak_array-peak_mean)/peak_mean*100.,marker='o',color='red')
            ax3.set_ylabel('PEAK' +'\n'+'res % [keV]')
            # ax3.set_yticks([peak_mean, peak_mean-0.0005*peak_mean, peak_mean+0.0005*peak_mean, \
            #                 peak_mean-0.00025*peak_mean, peak_mean+0.00025*peak_mean])
            ax3.axhline(y=0,color='black',label="%.3f"%peak_mean+' keV'+'\n'+'rms= '+"%.5f"%peak_rms,linewidth=3)
            ax3.grid()
            plt.xticks(start_array,time_array,rotation=90,size=10)
            ax3.set_xlabel='time'
            plt.subplots_adjust(wspace=0.01,hspace=0.2,top=0.975,bottom=0.25)
            ax3.legend()

            


            
        if splitted[0]=='SAVE':
            np.save(base_path+'counts.npy',counts_array)
            np.save(base_path+'dead.npy',dead_array)
            np.save(base_path+'start.npy',start_array)
            np.save(base_path+'livetime.npy',livetime_array)
            np.save(base_path+'norm.npy',norm_array)
            np.save(base_path+'peak.npy',peak_array)
            np.save(base_path+'sigma.npy',sigma_array)

      

   

       

            

if __name__ == "__main__":


    import argparse
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser( prog='calibrator.py',  formatter_class=formatter)
    parser.add_argument('infile', type=str, help='imput file')
    args = parser.parse_args()
    print('input file=',args.infile)
    #check file exist:
    if not (ospath.exists(args.infile)):
        print ("file not found:",args.infile)
        exit()
        
    plotAllSpectra(args.infile)
   

   
    plt.show()



