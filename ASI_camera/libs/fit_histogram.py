import numpy as np
from scipy.optimize import curve_fit


def linear_model(x,p0,p1):
       y=p0+p1*x
       return y


def gaussian_model(x,p0,p1,p2):
       #x, pars sono numpy array!!
     
       amplitude=p0
       peak=p1
       sigma=p2
       y=  amplitude * np.exp(-0.5 * ((x - peak)**2. / sigma**2.))
          
       return y

def get_centers(bins):
    centers=np.empty(0)
    for i in range(0,len(bins)-1):
        c=bins[i]+0.5*(bins[i+1]-bins[i])
        centers=np.append(centers,c)

    #print("centers=",centers)    
    return centers     
   
def fit_Gaushistogram(counts,bins,xmin=-100000,xmax=100000, initial_pars=[1,1,1]):
    bin_centers =get_centers(bins)          
    _mask = (counts > 0)&(bin_centers>xmin)&(bin_centers<xmax)
    y_data=counts[_mask]
    x_data=bin_centers[_mask]

    popt, pcov = curve_fit(gaussian_model, x_data, y_data,p0=initial_pars)
    
    return popt,  pcov




def fit_Gaushistogram_iterative(counts,bins,xmin=-100000,xmax=100000, initial_pars=[1,1,1], nSigma=1.5 ):

    popt, pcov =fit_Gaushistogram(counts,bins,xmin,xmax, initial_pars)
    k=popt[0]
    mean=popt[1]
    sigma=popt[2]
    xmin=mean-nSigma*sigma
    xmax=mean+nSigma*sigma
    for jj in range (0,2):
           
        xmin=mean-nSigma*sigma
        xmax=mean+nSigma*sigma  
        popt, pcov=    fit_Gaushistogram(counts,bins,xmin,xmax, initial_pars=[k,mean,sigma])
        k=popt[0]
        mean=popt[1]
        sigma=popt[2]

    return popt,  pcov, xmin,xmax
    
