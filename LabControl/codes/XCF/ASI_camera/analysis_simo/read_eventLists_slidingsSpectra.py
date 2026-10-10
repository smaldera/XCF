import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo


####
# small scripts to plot data from the event list con tagli
# plotta: spettro e mappa posizioni
# 

fileListName='events_file_list.txt'
#fileListName='/home/maldera/Desktop/eXTP/data/ASI_newSphere/Ge_111/3nov2023/event_listsPd_POL.txt'

ff=open(fileListName,'r')

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144
REBINXY=20.
xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

calP0=-0.003201340833319255
calP1=0.003213272145961988

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])

for f in ff:
    print(f)
    w, x,y=al.retrive_vectors(f[:-1])
    print(w)
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
  

#myCut=np.where( (w_all>2390)&(w_all<2393)  )
#myCut=np.where( (x_all>800)&(x_all<1200)&(y_all>1900)&(y_all<2500)  )
#myCut=np.where( (w_all>600)&(w_all<900)  )



y0=2000.
r=100.
#x0s=np.linspace(r,XBINS-r,num=int(XBINS/(2.*r)) )
x0s=np.linspace(r,XBINS-r,num=int(XBINS/(2.*r)) )
y0s=np.linspace(r,YBINS-r,num=int(YBINS/(2.*r)) )


low=2.55
up=2.85
k0=1000
mean0=0.5*(up+low)

fitted_mean=np.empty(0)
fitted_meanErr=np.empty(0)


#for x0 in x0s:
for y0 in y0s:

    #print ("xO=",x0)
    print ("yO=",y0)
    
#    myCut=np.where( (w_all>100)&( (x_all-x0)**2+(y_all-y0)**2<r**2 ) ) # cerchi
    #myCut=np.where( (w_all>100)&(y_all>x0-r+10)&(x_all<x0+r-10)&(y_all>1500)&(y_all<3000)   ) # rettangoli
    myCut=np.where( (w_all>100)&(y_all>y0-r)&(y_all<y0+r)&(x_all>0)&(x_all<XBINS)   ) # rettangoli
  
    
    #plot 
    counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
    counts2dClu=   counts2dClu.T
    plt.figure(1)
    plt.imshow(np.log10(counts2dClu), interpolation='none', origin='upper',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0.1, vmax=1.6)
    #  plt.colorbar()

    plt.figure(2)
    countsClu, bins = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
    #apply energy calibration
    bins=bins*calP1+calP0

    # fit histogram:
    popt,  pcov, xmin,xmax, redChi2= fitSimo.fit_Gaushistogram_iterative(countsClu,bins,xmin=low,xmax=up, initial_pars=[k0,mean0,10], nSigma=1.1 )

    mean=popt[1]
    meanErr=pcov[1][1]

    fitted_mean=np.append(mean,fitted_mean)
    fitted_meanErr=np.append(meanErr**0.5,fitted_meanErr)
    
    
    #print("fitted mean=",mean, " err=",meanErr, " reduced chi2=",redChi2)
    #x=np.linspace(xmin,xmax,1000)
    #y= fitSimo.gaussian_model(x,popt[0],popt[1],popt[2])
    #plt.plot(x,y,'r-')
    #plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="x0="+str(x0))
   
    plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="y0="+str(y0))
    
    
plt.xlabel('E [keV]')
plt.legend()


plt.figure(3)
#plt.error(x0s,fitted_mean,'or')
plt.errorbar(x0s,fitted_mean,yerr=fitted_meanErr, xerr=r-10, fmt='ro')
plt.xlabel('Xbin')
plt.ylabel('meanE [keV]')

plt.show()
