import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
from cutHotPixels import hotPixels

from scipy.optimize import curve_fit




def fit_peak(hSpec,  min_x1, max_x1,   amplitude,    peak,   sigma,n_sigma=1):
         

    par, cov, xmin,xmax, chi2 = fitSimo.fit_Gaushistogram_iterative(hSpec.counts, hSpec.bins, xmin=min_x1,xmax=max_x1, initial_pars=[amplitude,peak,sigma],  nSigma=n_sigma )
    print(' ')
    print('FIT PARAMETERS')
    print('Gaussian norm = ', "%.5f"%par[0],' +- ',"%.5f"%np.sqrt(cov[0][0]),' keV')
    print('Gaussian peak = ', "%.5f"%par[1],' +- ',"%.5f"%np.sqrt(cov[1][1]),' keV')
    print('Gaussian sigma = ', "%.5f"%par[2],' +- ',"%.5f"%np.sqrt(cov[2][2]),' keV')
    print(' ')

    return   par, cov, xmin,xmax, chi2
    
def draw_and_save( w_i,x_i, y_i,DIR,suffix):
    
         fig=plt.figure(figsize=(10,10))
         ax1=plt.subplot(111)
         NBINS=int(16384) 
         #plot 
         # mappa posizioni:
         counts2dClu,  xedges, yedges= np.histogram2d(x_i,y_i,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
         counts2dClu=   counts2dClu.T
         im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='upper',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
         ax1.set_xlabel('X')
         ax1.set_ylabel('Y')
         plt.colorbar(im,ax=ax1)

         #ax2=plt.subplot(312)
         fig2=plt.figure(figsize=(10,10))
         ax2=plt.subplot(111)
         
         calP1= 0.0032132721459619882
         calP0=-0.003201340833319255
         # spettro energia
         #plt.figure(2)
         #countsClu, binsE = np.histogram( w_i*calP1+calP0  , bins = NBINS, range = (0,16384*calP1+calP0) )
         countsClu, binsE = np.histogram( w_i, bins = NBINS, range = (0,NBINS) )
         binsE=binsE*calP1+calP0       
       
         ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
         ax2.legend()
         ax2.set_xlabel('E[keV]')
         ax2.set_xlim([0,10])
         ax2.set_yscale('log')
         
                  
         if SAVE_HISTOGRAMS==True:
             spectrum_file_name =DIR + '/spectrumPos_'+suffix+'.npz'
             np.savez(spectrum_file_name, counts = countsClu,  bins = binsE)
             fig.savefig(DIR+'/img_'+suffix+'.png')
             
                         
########################################################################

def draw_and_recalibrate( w_i,x_i, y_i,DIR,suffix):
    
         fig=plt.figure(figsize=(10,10))
         ax1=plt.subplot(111)
         NBINS=int(16384) 
         #plot 
         # mappa posizioni:
         counts2dClu,  xedges, yedges= np.histogram2d(x_i,y_i,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
         counts2dClu=   counts2dClu.T
         im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='upper',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
         ax1.set_xlabel('X')
         ax1.set_ylabel('Y')
         plt.colorbar(im,ax=ax1)

         #ax2=plt.subplot(312)
         fig2=plt.figure(figsize=(10,10))
         ax2=plt.subplot(111)
         
         calP1= 0.0032132721459619882
         calP0=-0.003201340833319255
         # spettro energia
         #plt.figure(2)
         #countsClu, binsE = np.histogram( w_i*calP1+calP0  , bins = NBINS, range = (0,16384*calP1+calP0) )
         countsClu, binsE = np.histogram( w_i, bins = NBINS, range = (0,NBINS) )
         binsE=binsE*calP1+calP0

         hSpec=histogramSimo()
         hSpec.counts=countsClu
         hSpec.bins=binsE

       
         # fit Lalpha
         min_x1=2.26
         max_x1=2.34
         #amplitude=100.          peak=2.29          sigma=0.5
         par1, cov1, min_x1,max_x1, chi21 = fit_peak(hSpec,min_x1 ,max_x1,  1e5,   2.29,  0.1,n_sigma=0.7)

         
         # fit LBeta
         min_x2=2.37
         max_x2=2.43
         #amplitude=100.          peak=2.29          sigma=0.5
         par2, cov2, min_x2,max_x2, chi22 = fit_peak(hSpec,min_x2 ,max_x2,  1e5,   2.40,  0.1,n_sigma=0.4)
         
         # fit Si
         min_x3=1.7
         max_x3=1.78
         #amplitude=100.          peak=2.29          sigma=0.5
         par3, cov3, min_x3,max_x3, chi23 = fit_peak(hSpec,min_x3 ,max_x3,  1e5,   1.74,  0.1,n_sigma=0.7)
         
         # fit escape La
         min_x4=0.52
         max_x4=0.58
         #amplitude=100.          peak=2.29          sigma=0.5
         par4, cov4, min_x4,max_x4, chi24 = fit_peak(hSpec,min_x4 ,max_x4,  1e5,   0.55,  0.1,n_sigma=0.7)

         # fit escape Lb
         min_x5=0.62
         max_x5=0.68
         #amplitude=100.          peak=2.29          sigma=0.5
         par5, cov5, min_x5,max_x5, chi25 = fit_peak(hSpec,min_x5 ,max_x5,  1e5,   0.65,  0.1,n_sigma=0.4)

         
         

         
         ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
         x=np.linspace(min_x1,max_x1,1000) 
         plt.plot(x,fitSimo.gaussian_model(x,par1[0],par1[1],par1[2]),label='fit kalpha')

         x2=np.linspace(min_x2,max_x2,1000) 
         plt.plot(x2,fitSimo.gaussian_model(x2,par2[0],par2[1],par2[2]),label='fit Kbeta')

         x3=np.linspace(min_x3,max_x3,1000) 
         plt.plot(x3,fitSimo.gaussian_model(x3,par3[0],par3[1],par3[2]),label='fit Si')

         x4=np.linspace(min_x4,max_x4,1000) 
         plt.plot(x4,fitSimo.gaussian_model(x4,par4[0],par4[1],par4[2]),label='fit La escape')

         x5=np.linspace(min_x5,max_x5,1000) 
         plt.plot(x5,fitSimo.gaussian_model(x5,par5[0],par5[1],par5[2]),label='fit Lb escape')
        
                 
    


         meanLa=par1[1]
         meanLb=par2[1]
         ELa=2.2932
         ELb=2.3948
         ESi=1.74
         ELa_escape=ELa-ESi
         ELb_escape=ELb-ESi

         #ricalibrazione energia
        
         true= np.array([ELa_escape, ELb_escape,ESi,ELa,ELb])
         fitted_mean=np.array([par4[1],par5[1],par3[1],par1[1],par2[1]])
         fitted_meanErr=np.array([cov4[1][1]**0.5,cov5[1][1]**0.5,cov3[1][1]**0.5,cov1[1][1]**0.5,cov2[2][1]**0.5])
         poptCal, pcovCal = curve_fit(fitSimo.linear_model, true, fitted_mean,p0=[0,1], absolute_sigma=True, sigma=fitted_meanErr, bounds=(-np.inf, np.inf )   )
         chisq = (((fitted_mean - fitSimo.linear_model(true,poptCal[0],poptCal[1]))/fitted_meanErr)**2).sum()
         ndof= len(true) - len(poptCal)
         redChi2=chisq/ndof
         print('chi2=',chisq," ndof=",ndof, " chi2/ndof=",redChi2)

         
         calP1corr= poptCal[1] 
         calP0corr= poptCal[0] 

         print("calP1corr=",calP1corr, "  calP0corr=", calP0corr)

         countsCluCorr, binsE = np.histogram(( w_i*calP1+calP0)*calP1corr+calP0corr  , bins = NBINS, range = (0,(16384-1)*calP1+calP0) )
        # ax2.hist(binsE[:-1], bins = binsE, weights = countsCluCorr, histtype = 'step',label="energy w. clustering corrCalib")
         ax2.legend()
         ax2.set_xlabel('E[keV]')
         ax2.set_xlim([0,10])
         ax2.set_yscale('log')

         # plot energyCalib
         fig3=plt.figure(figsize=(10,10))
         ax3=plt.subplot(111)

         ax3.plot(fitted_mean,true,'bo')
         x=np.linspace(0.1,5,100)
         ax3.plot(x,x*  calP1corr+   calP0corr,'-r')
                  
         if SAVE_HISTOGRAMS==True:
            # DIR = args.saveDir
             spectrum_file_name =DIR + '/spectrumCorrPos_'+suffix+'.npz'
             print('... saving energy spectrun  in:', spectrum_file_name  )
             #np.savez(spectrum_file_name, counts = countsCluCorr,  bins = binsE)
             spectrum_file_name =DIR + '/spectrumPos_'+suffix+'.npz'
             np.savez(spectrum_file_name, counts = countsClu,  bins = binsE)
           
             fig.savefig(DIR+'/img_'+suffix+'.png')
             
             with open(DIR+'/corrCalib_'+suffix+'.txt', 'w') as f:
                 f.write("calP1corr="+str(calP1corr)+" calP0corr="+str(calP0corr)+'\n')
                 f.write("normLa="+str(par1[0])+" normLaErr="+str(cov1[0][0]**0.5)+ " meanLa="+str(par1[1])+" sigmaLa="+ str(par1[2])+'\n')
                 f.write("normLb="+str(par2[0])+" normLbErr="+str(cov2[0][0]**0.5)+" meanLb="+str(par2[1])+" sigmaLb="+str(par2[2])+'\n' )
                 f.write("normSi="+str(par3[0])+" normSiErr="+str(cov3[0][0]**0.5)+" meanSi="+str(par3[1])+" sigmaSi="+str(par3[2])+'\n' )
                 f.write("normLa_esc="+str(par4[0])+" normLa_EscErr="+str(cov4[0][0]**0.5)+" meanLa_esc="+str(par4[1])+" sigmaLa_esc="+str(par4[2])+'\n' )
                 f.write("normLb_esc="+str(par5[0])+" normLb_escErr="+str(cov5[0][0]**0.5)+" meanLb_esc="+str(par5[1])+" sigmaLb_esc="+str(par5[2])+'\n')
                 
             f.close()    
             
########################################################################
             
import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-in','--inFile', type=str,  help='txt file with list of npz files', required=True)
parser.add_argument('-dir','--saveDir', type=str,  help='direxctory where npz files are saved', required=False)

args = parser.parse_args()

FIND_HOTPIXELS=False
CUT_HOT_PIXELS=False


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
#####################
#selezione hot pixels

if FIND_HOTPIXELS==True:
    hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all)
    hotPix.find_HotPixels(n_sigma=10,low_threshold=10)
    hotPix.save_cuts(DIR+'/cuts.npz')
if CUT_HOT_PIXELS==True:
    hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all)
    hotPix.retrive_cuts(DIR+'/cuts.npz')
    hotPix.applyCuts()
    w_all,   x_all,  y_all, size_all=hotPix.get_cutVectors()
                    

print("len w_all dopo cut ",len(w_all))

#################3

x_inf0=250
x_sup0=2820
y_inf0=1000
y_sup0=3500
myCut0=np.where( (w_all>100)&(y_all>y_inf0)&(y_all<y_sup0)&(x_all>x_inf0)&(x_all<x_sup0))


n_events=len(w_all[myCut0])
events_bins=2
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

         suffix=str(i)+'_'+str(j)
         #draw_and_recalibrate( w_i,x_i, y_i,args.saveDir,suffix)
         draw_and_save( w_i,x_i, y_i,args.saveDir,suffix)


# spettro total:
w_i=w_all[myCut0]
x_i=x_all[myCut0]
y_i=y_all[myCut0]
size_i=size_all[myCut0]
#draw_and_recalibrate( w_i,x_i, y_i,args.saveDir,'all')
draw_and_save( w_i,x_i, y_i,args.saveDir,'all')


plt.show()
