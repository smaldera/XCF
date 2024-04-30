import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo


####
# small scripts to plot CMOS data from the event list whit cuts
# draws: 2D map,  energy, x-y projections
# 


def fit_peak(hspectrum,  min_x1, max_x1,   amplitude,    peak,   sigma,n_sigma=1):
         
    
    #par, cov, chi2 = fitSimo.fit_Gaushistogram(hSpec.counts, hSpec.bins, xmin=min_x1,xmax=max_x1, initial_pars=[amplitude,peak,sigma], parsBoundsLow=-np.inf, parsBoundsUp=np.inf )
    par, cov, xmin,xmax, chi2 = fitSimo.fit_Gaushistogram_iterative(hSpec.counts, hSpec.bins, xmin=min_x1,xmax=max_x1, initial_pars=[amplitude,peak,sigma],  nSigma=n_sigma )
    print(' ')
    print('FIT PARAMETERS')
    print('Gaussian norm = ', "%.5f"%par[0],' +- ',"%.5f"%np.sqrt(cov[0][0]),' keV')
    print('Gaussian peak = ', "%.5f"%par[1],' +- ',"%.5f"%np.sqrt(cov[1][1]),' keV')
    print('Gaussian sigma = ', "%.5f"%par[2],' +- ',"%.5f"%np.sqrt(cov[2][2]),' keV')
    print(' ')

    return   par, cov, xmin,xmax, chi2
    
    



import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-in','--inFile', type=str,  help='txt file with list of npz files', required=True)
parser.add_argument('-dir','--saveDir', type=str,  help='direxctory where npz files are saved', required=False)

args = parser.parse_args()




ff=open(args.inFile,'r')


# retta calibrazione cmos
calP1= 0.0032132721459619882
calP0=-0.003201340833319255

NBINS=int(16384)  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=1.

SAVE_HISTOGRAMS=True
spectrum_file_name='test_spectrum.npz'
xproj_file_name='test_xproj.npz'
yproj_file_name='test_yproj.npz'


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])
size_all=np.array([])

for f in ff:
    print(f)
    w, x,y,size=al.retrive_vectors2(f[:-1])
    print(w)
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
    size_all=np.append(size_all,size)

# CUT di SELEZIONE EVENTI!!!

x_inf0=250
x_sup0=2820
y_inf0=1000
y_sup0=3500
myCut0=np.where( (w_all>100)&(y_all>y_inf0)&(y_all<y_sup0)&(x_all>x_inf0)&(x_all<x_sup0))


n_events=len(w_all[myCut0])
events_bins=1.
print("n_eventsAll=",n_events)

deltaX=(x_sup0-x_inf0)/events_bins
deltaY=(y_sup0-y_inf0)/events_bins

for i in range(0, int(events_bins)):
   x_inf=x_inf0+i*deltaX 
   x_sup=x_inf0+(i+1)*deltaX 
   for j in range(0, int(events_bins)):
         y_inf=y_inf0+j*deltaY
         y_sup=y_inf0+(j+1)*deltaY

         print("i=",i," j=",j," x_inf=",x_inf," x_sup=",x_sup," y_sup=",y_sup,' y_inf=',y_inf)
         myCut=np.where( (w_all>100)&(y_all>y_inf)&(y_all<y_sup)&(x_all>x_inf)&(x_all<x_sup))
         w_i=w_all[myCut]
         x_i=x_all[myCut]
         y_i=y_all[myCut]
         size_i=size_all[myCut]


         fig=plt.figure(figsize=(10,10))
         ax1=plt.subplot(211)

         #plot 
         # mappa posizioni:
         counts2dClu,  xedges, yedges= np.histogram2d(x_i,y_i,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
         counts2dClu=   counts2dClu.T
         im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='upper',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
         ax1.set_xlabel('X')
         ax1.set_ylabel('Y')
         plt.colorbar(im,ax=ax1)

         ax2=plt.subplot(212)
         # spettro energia
         #plt.figure(2)
         countsClu, binsE = np.histogram( w_i  , bins = 2*NBINS, range = (-16384,16384) )
         binsE=binsE*calP1+calP0

         hSpec=histogramSimo()
         hSpec.counts=countsClu
         hSpec.bins=binsE

         # fit ka
         min_x1=2.26
         max_x1=2.34
         #amplitude=100.          peak=2.29          sigma=0.5
         par1, cov1, min_x1,max_x1, chi21 = fit_peak(hSpec,min_x1 ,max_x1,  1e5,   2.29,  0.1,n_sigma=0.7)
         
         # fit kBeta
         min_x2=2.37
         max_x2=2.43
         #amplitude=100.          peak=2.29          sigma=0.5
         par2, cov2, min_x2,max_x2, chi22 = fit_peak(hSpec,min_x2 ,max_x2,  1e5,   2.40,  0.1,n_sigma=0.4)
         

         
         

         
         ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
         x=np.linspace(min_x1,max_x1,1000) 
         plt.plot(x,fitSimo.gaussian_model(x,par1[0],par1[1],par1[2]),label='fit kalpha')

         x2=np.linspace(min_x2,max_x2,1000) 
         plt.plot(x2,fitSimo.gaussian_model(x2,par2[0],par2[1],par2[2]),label='fit Kbeta')

         ax2.set_xlabel('E[keV]')
         ax2.set_xlim([0,10])
         ax2.set_yscale('log')
         ax2.legend()

         if SAVE_HISTOGRAMS==True:
             DIR = args.saveDir
             print('... saving energy spectrun  in:', spectrum_file_name  )
             np.savez(DIR + 'spectrumPos_'+str(i)+'_'+str(j)+'.npz', counts = countsClu,  bins = binsE)
             fig.savefig(DIR+'img_'+str(i)+'_'+str(j)+'.png')


# spettro total:
countsClu_all, binsE = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-16384,16384) )
binsE=binsE*calP1+calP0

np.savez(DIR + 'spectrum_all.npz', counts = countsClu,  bins = binsE)


plt.show()
