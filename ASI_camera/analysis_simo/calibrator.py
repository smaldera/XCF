import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, '../libs')
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
    meanErr=pcov[1][1]
    print("fitted mean=",mean, " err=",meanErr, " reduced chi2=",redChi2)
    

    #plot?
    if plot:
        x=np.linspace(xmin,xmax,1000)
        y= fitSimo.gaussian_model(x,popt[0],popt[1],popt[2])
        plt.plot(x,y,'r-',label='fitted function')

    
    return  mean, meanErr



def calibrator(calibFileName):

    f=open(calibFileName)
    p=histogramSimo()
    n_spectra=0
    
    true=[]
    fitted_mean=[]
    fitted_meanErr=[]
    fig, ax = plt.subplots()
    for line in f:
        line=line.strip('\n')
       # print(line)
        if len(line)==0:
            continue
        if line[0]=='#':
            continue
        splitted=line.split('=')
        if splitted[0]=='FILE':
            print("reading file",splitted[1])
            n_spectra+=1

            data_array, deadTime, livetime, fast_counts =pharse_mca(splitted[1])
            size=len(data_array)      
            bin_edges=np.linspace(0,size+1,size+1)
            p.counts=data_array
            p.bins=bin_edges
            # plot istogramma?
            #fig, ax = plt.subplots()
            p.plot(ax,splitted[1].split('/')[-1])
            plt.legend()

        if splitted[0]=='PEAK':
            peak_string=splitted[1].split(' ')
            true_val=float(peak_string[0])
            low=float(peak_string[1])
            up=float(peak_string[2])
            print("true=",true_val,"  low=",low," up = ",up)
            true.append(true_val) 
            mean,meanErr=fit_histo(p,low,up)
            fitted_mean.append(mean)
            fitted_meanErr.append(meanErr)


    return      np.array(true),   np.array(fitted_mean),  np.array(fitted_meanErr )

       

            

if __name__ == "__main__":


    true,   fitted_mean,  fitted_meanErr = calibrator('calibrator_input.txt')
    
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
     
    plt.show()



