import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os.path as ospath

import sys
#sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo



def fit_histo(p,low,up, plot=1):

    k0=100
    mean0=low+(up-low)/2.
    counts=p.counts
    bins=p.bins     
    popt,  pcov, xmin,xmax, redChi2= fitSimo.fit_Gaushistogram_iterative(counts,bins,xmin=low,xmax=up, initial_pars=[k0,mean0,10], nSigma=1.1 )
    mean=popt[1]
    meanErr=pcov[1][1]**0.5
    sigma=popt[2]
    sigmaErr=pcov[2][2]**0.5
    print("fitted mean=",mean, " err=",meanErr, " reduced chi2=",redChi2)

    
    #plot?
    if plot:
        x=np.linspace(xmin,xmax,1000)
        y= fitSimo.gaussian_model(x,popt[0],popt[1],popt[2])
        plt.plot(x,y,'r-')

               
        
    
    return  mean, meanErr, sigma, sigmaErr



def calibrator(calibFileName):

    f=open(calibFileName)
    p=histogramSimo()
    n_spectra=0
    base_path=''
    
    true=[]
    fitted_mean=[]
    fitted_meanErr=[]
    fitted_sigma=[]
    fitted_sigmaErr=[]
    
    fig, ax = plt.subplots()
    for line in f:
        line=line.strip('\n')
       # print(line)
        if len(line)==0:
            continue
        if line[0]=='#':
            continue
        splitted=line.split('=')
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
            n_spectra+=1
            
            p.read_from_file(filename, fileFormat )
            
            # plot istogramma?
            #fig, ax = plt.subplots()
            p.plot(ax,filename.split('/')[-1])
            plt.legend()
          

        if splitted[0]=='PEAK':
            peak_string=splitted[1].split(' ')
            true_val=float(peak_string[0])
            low=float(peak_string[1])
            up=float(peak_string[2])
            print("true=",true_val,"  low=",low," up = ",up)
            true.append(true_val) 
            mean,meanErr,sigma,sigmaErr=fit_histo(p,low,up)
            fitted_mean.append(mean)
            fitted_meanErr.append(meanErr)
            fitted_sigma.append(sigma)
            fitted_sigmaErr.append(sigmaErr)
          
            
    return      np.array(true),   np.array(fitted_mean),  np.array(fitted_meanErr ),  np.array(fitted_sigma),  np.array(fitted_sigmaErr ) 

       

            

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
        
    true,   fitted_mean,  fitted_meanErr, fitted_sigma,  fitted_sigmaErr  = calibrator(args.infile)
   

    n_files=3
    plt.figure(n_files+1)
    plt.errorbar(true, fitted_mean ,yerr=fitted_meanErr, fmt='ro')

    # fit retta calib:
    
    poptCal, pcovCal = curve_fit(fitSimo.linear_model, true, fitted_mean ,absolute_sigma=True, sigma=fitted_meanErr, bounds=(-np.inf, np.inf )   )
    chisq = (((fitted_mean - fitSimo.linear_model(true,poptCal[0],poptCal[1]))/fitted_meanErr)**2).sum()
    ndof= len(true) - len(poptCal)
    redChi2=chisq/ndof
    print('chi2=',chisq," ndof=",ndof, " chi2/ndof=",redChi2)


    x=np.linspace(min(true)-2 ,max(true)+2,8000)
    y= fitSimo.linear_model(x,poptCal[0],poptCal[1])
    plt.plot(x,y,'b-')

    print ("cal parameters=",poptCal)
    print("cov matrix=",pcovCal)
   
    p0=poptCal[0]
    p1=poptCal[1]

    p0Err=(pcovCal[0][0])**0.5
    p1Err=(pcovCal[1][1])**0.5
    p0p1Cov=pcovCal[0][1]
  
    print("p0=",p0," p0Err=",p0Err," p1=",p1,"p1Err=",p1Err," covp0p1=",p0p1Cov)

    calP0=-p0/p1
    calP1=1./p1
    calP1Err=((1./p1)**2)*p1Err
    calP0Err=( ((1./p1)*p0Err)**2+  (p1Err*p0/(p1**2))**2  )**0.5
    
    


    print("CAL P0=",calP0," calP0Err=",calP0Err,"  calP1= ",calP1, "  calP1Err=",calP1Err)
    plt.figure(n_files+2)
    plt.errorbar(fitted_mean,true ,xerr=fitted_meanErr, fmt='ro')
    x=np.linspace(min(fitted_mean)-2000 ,max(fitted_mean)+2000,1000)
    y= fitSimo.linear_model(x,calP0,calP1)
    y_err=y+3*((x*calP1Err)**2+calP0Err**2)**0.5
    plt.plot(x,y,'b-')
    plt.plot(x,y_err,'k-')
     

    
    plt.figure(10)
    plt.errorbar(true,100.*fitted_sigma/fitted_mean ,yerr=100.*fitted_sigmaErr/fitted_mean, fmt='ro') 
    x=np.linspace(0,10,1000)
    y= 100.*(0.1*3.6/(1000.*x))**.5
    plt.plot(x,y,'-')

    plt.show()



