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
peak_en_arr, sigma_en_arr = [], []
peak_en_err_arr, sigma_en_err_arr = [], []
names_arr = []

def EtoADC(E,P0,P1):
    return (E-P0)/P1

def ADCtoE(adc,P0,P1):
    return adc*P1+P0

def ADCtoE_err(adc,P0,P1,adc_err,P0_err,P1_err):
    sigma_ = P1**2*adc_err**2+adc**2*P1_err**2+P0_err**2
    return np.sqrt(sigma_)


def  plotAllSpectra(InputFileName):

    f=open(InputFileName)
    p=histogramSimo()
    p_adc=histogramSimo()
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
            p_adc.read_from_file(filename, fileFormat )

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
            compute_rate=1
            if fileFormat=='sdd' or  fileFormat=='sddnpz':
                time=p.sdd_liveTime
                print("livetime=",time)
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
            # p.normalize(low,up)
            normalize=1 
            

                
 
        if splitted[0]=="ADD_PLOT": 
            # plot istogramma?
            if fileFormat=='sdd' or  fileFormat=='sddnpz':
                calP0=-0.03544731540487446
                calP1=0.0015013787118821926
                calP0_err = 0.0004
                calP1_err = 9.6e-8
            if fileFormat=='npz':
                calP1=0.0032132721459619882
                calP0=-0.003201340833319255
                calP0_err = 2.8e-5
                calP1_err = 2.8e-8
            if P0_ != None:
                calP0 = P0_
            if P1_ != None:
                calP1 = P1_
            
            print("calP1=",calP1,"  calP0=",calP0)    
            p.bins=p.bins*calP1+calP0
            plt.ylabel('counts')
            
            if compute_rate==1:
                 print("compute rate time=",time)
                 p.counts=p.counts/time
                 p_adc.counts=p_adc.counts/time
                 compute_rate=0
                 time=1
                 plt.ylabel('counts/s')
            if normalize==1:
                 print("NORMALIZZO!!!!!")
                 p.normalize(low,up)
                 low_adc = EtoADC(low,calP0,calP1)
                 up_adc = EtoADC(up,calP0,calP1)
                 Max = p_adc.max_in_spectrum(low_adc,up_adc)
                #  p_adc.normalize(low_adc,up_adc)
                 normalize=0
                 low=0.
                 up=0.
                 plt.ylabel('normalized rate')
                 #p_adc.plot(ax,None)
            if show_legend==1:
                p.plot(ax,legend)
            else:
                p.plot(ax,None)
            plt.xlabel('energy [keV]')
            
           
          #  plt.xlim(0.5,6)
          #  plt.ylim(7e-3,1.1)
            plt.yscale('log')
            plt.legend()


        
        if splitted[0]=="FIT":    
            fit_parameter=splitted[1].split(' ')
            
            min_x = EtoADC(float(fit_parameter[0]),calP0,calP1)
            max_x = EtoADC(float(fit_parameter[1]),calP0,calP1)
            amplitude = float(fit_parameter[2])
            peak = EtoADC(float(fit_parameter[3]),calP0,calP1)
            sigma = EtoADC(float(fit_parameter[4]),calP0,calP1)
            mask_adc_fit = np.where((p_adc.bins >= min_x) & (p_adc.bins <= max_x))[0]
            entries_in_fit_range = len(p_adc.counts[mask_adc_fit])
            par, cov, chi2 = fitSimo.fit_Gaushistogram(p_adc.counts, p_adc.bins, xmin=min_x,xmax=max_x, initial_pars=[amplitude,peak,sigma], parsBoundsLow=-np.inf, parsBoundsUp=np.inf )
            x=np.linspace(min_x,max_x,1000)
            peak_En = ADCtoE(par[1],calP0,calP1)
            peak_En_p1sigma = ADCtoE(par[1]+par[2],calP0,calP1)
            peak_En_m1sigma = ADCtoE(par[1]-par[2],calP0,calP1)
            sigma_En = (peak_En_p1sigma-peak_En_m1sigma)/2 #ADCtoE(par[2],calP0,calP1)
            peak_err_En = ADCtoE(np.sqrt(cov[1][1]),calP0,calP1)
            sigma_err_En = ADCtoE(np.sqrt(cov[2][2]),calP0,calP1)
            peak_err_tot_En = ADCtoE_err(par[1],calP0,calP1,np.sqrt(cov[1][1]),calP0_err,calP1_err)
            sigma_err_tot_En = ADCtoE_err(par[2],calP0,calP1,np.sqrt(cov[2][2]),calP0_err,calP1_err)
            # breakpoint()
            CMOS_res = True
            if CMOS_res:
                def resolution(peaks,sigma,peaks_err,sigma_err):
                    FWHM = 2.355*sigma
                    FWHM_err = 2.355*sigma_err
                    res = FWHM/peaks
                    res_err = np.sqrt((FWHM_err / peaks)**2 + (res * peaks_err / peaks)**2)
                    return res, res_err

                res, res_err = resolution(peak_En,sigma_En,peak_err_En,sigma_err_En)
                res_adc, res_adc_err = resolution(par[1],par[2],np.sqrt(cov[1][1]),np.sqrt(cov[2][2]))
            if show_legend==1:
                if not CMOS_res:
                    label=f'E = {np.round(peak_En,4)}'+r'$\pm$' + f'{np.round(peak_err_tot_En,4)} keV'+'\n'+r'$\sigma$'+f' = ({np.round(sigma_En*100,3)}'+r'$\pm$' + f'{np.round(sigma_err_tot_En*100,3)})'+r'$\times10^{-2}$ keV'
                else:
                    label=f'E = {np.round(peak_En,4)}'+r'$\pm$' + f'{np.round(peak_err_tot_En,4)} keV'+'\n'+r'$\sigma$'+f' = ({np.round(sigma_En*100,3)}'+r'$\pm$' + f'{np.round(sigma_err_tot_En*100,3)})'+r'$\times10^{-2}$ keV'+'\n'+r'$\Delta E/E$ ='+ f'{np.round(res*100.,1)}'+r'$\pm$' + f'{np.round(res_err*100.,1)} %'
            else:
                label=None
            xx = np.linspace(peak_En-sigma_En,peak_En+sigma_En,1000)
            ax.plot(xx,fitSimo.gaussian_model(xx,par[0]/np.max(p_adc.counts),peak_En,sigma_En),label=label)
            # xx_ = np.linspace(par[1]-3*par[2],par[1]+3*par[2],1000)
            # ax.plot(xx_,fitSimo.gaussian_model(xx_,par[0],par[1],par[2]),label=label)
            print(' ')
            print('FIT PARAMETERS')
            print('Gaussian norm = ', "%.5f"%par[0],' +- ',"%.5f"%np.sqrt(cov[0][0]))
            print('Gaussian peak ADC = ', "%.5f"%par[1],' +- ',"%.5f"%np.sqrt(cov[1][1]))
            print('Gaussian peak = ', "%.5f"%peak_En,' +- ',"%.5f"%peak_err_En,' err tot =',"%.5f"%peak_err_tot_En,' keV')
            print('Gaussian sigma ADC = ', "%.5f"%par[2],' +- ',"%.5f"%np.sqrt(cov[2][2]))
            print('Gaussian sigma = ', "%.5f"%sigma_En,' +- ',"%.5f"%sigma_err_En,' err tot =',"%.5f"%sigma_err_tot_En,' keV')
            if CMOS_res:
                print(f'res = {np.round(res*100.,2)} +- {np.round(res_err*100.,2)} %')
                print(f'ADC res = {np.round(res_adc*100.,2)} +- {np.round(res_adc_err*100.,2)} %')
            print(' ')
            print('         ####################################################################################')
            print('         ####################################################################################')

            # plt.xlabel('energy [keV]')
            
            # plt.ylabel('events/s') # non so perche', ,ma nell'if non funziona!

            plt.legend()

            peak_en_arr.append(peak_En)
            peak_en_err_arr.append(peak_err_tot_En)
            sigma_en_arr.append(sigma_En)
            sigma_en_err_arr.append(sigma_err_tot_En)
            names_arr.append(legend)

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
    parser.add_argument('--outpath', type=str, help='output path directory where to save .npy arrays containing the fit results',required=False,default=None)
    args = parser.parse_args()
    print('input file=',args.infile)
    out_path = args.outpath
    #check file exist:
    if not (ospath.exists(args.infile)):
        print ("file not found:",args.infile)
        exit()
        
    plotAllSpectra(args.infile)
   
    if out_path is not None:
        np.save(out_path+'peak_en.npy',peak_en_arr)
        np.save(out_path+'peak_en_err.npy',peak_en_err_arr)
        np.save(out_path+'sigma_en.npy',sigma_en_arr)
        np.save(out_path+'sigma_en_err.npy',sigma_en_err_arr)
        np.save(out_path+'names.npy',names_arr)
   
    plt.show()



