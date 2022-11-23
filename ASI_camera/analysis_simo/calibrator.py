import numpy as np
from matplotlib import pyplot as plt

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
            fig, ax = plt.subplots()
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


    return      true,   fitted_mean,  fitted_meanErr 

       

            

if __name__ == "__main__":


    true,   fitted_mean,  fitted_meanErr = calibrator('calibrator_input.txt')


    plt.figure(3)
    plt.plot(fitted_mean,true,'or--')
    
    plt.show()



