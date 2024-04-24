import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

# TODO = trattare bin vuoti
# errore poissoninano


def linear_model(x,p0,p1):
       y=p0+p1*x
       return y

def linear_model0(x,p1):
       y=p1*x
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
   
def fit_Gaushistogram(counts,bins,xmin=-100000,xmax=100000, initial_pars=[1,1,1], parsBoundsLow=-np.inf, parsBoundsUp=np.inf ):
    bin_centers =get_centers(bins)          
    _mask = (counts > 0)&(bin_centers>xmin)&(bin_centers<xmax)
    y_data=counts[_mask]
    x_data=bin_centers[_mask]
    sigma=np.sqrt(y_data)

    print("y_data=",y_data)
    print("len(y_data)=",len(y_data))
    
    print("Bounds=",(parsBoundsLow, parsBoundsUp ))
    
    popt, pcov = curve_fit(gaussian_model, x_data, y_data,p0=initial_pars,absolute_sigma=True, sigma=sigma, bounds=(parsBoundsLow, parsBoundsUp ), maxfev=5000)

    y_fit= gaussian_model(x_data,popt[0],popt[1],popt[2])

    chisq = (  ((y_data - y_fit)**2)/y_fit).sum()
    ndof= len(y_data) - len(popt)
    redChi2=chisq/ndof
    
    print("fitting histo, chi2=",chisq," ndof=",ndof,"  red chi2=",redChi2)
    
    return popt,  pcov,redChi2  




def fit_Gaushistogram_iterative(counts,bins,xmin=-100000,xmax=100000, initial_pars=[1,1,1], nSigma=1.5 ):
    myparsBoundsLow=[0,xmin-50,0]
    myparsBoundsUp=[np.inf,xmax+50,np.inf]
    #myparsBoundsLow=[0,-np.inf,0]
    #myparsBoundsUp=[np.inf,np.inf,np.inf]

    
    popt, pcov, redChi2 =fit_Gaushistogram(counts,bins,xmin,xmax, initial_pars,parsBoundsLow= myparsBoundsLow, parsBoundsUp= myparsBoundsUp  )
    k=popt[0]
    mean=popt[1]
    sigma=popt[2]
    xmin=mean-nSigma*sigma
    xmax=mean+nSigma*sigma
    for jj in range (0,5):
           
      #  xmin=min(mean-nSigma*sigma,initial_pars[1]-1 )
      #  xmax=max(mean+nSigma*sigma,initial_pars[1]+1)
      # print("xmin=",xmin, "xmax=",xmax)
        
        popt, pcov,  redChi2 =    fit_Gaushistogram(counts,bins,xmin,xmax, initial_pars=[k,mean,sigma],parsBoundsLow= myparsBoundsLow, parsBoundsUp= myparsBoundsUp )
        k=popt[0]
        mean=popt[1]
        sigma=popt[2]
        print("mean = ",mean," sigma=",sigma)

    print ("============>>>>>>>> Final  CHI2/ndof= ",redChi2)   
        
    return popt,  pcov, xmin,xmax, redChi2
    
