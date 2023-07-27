import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0,'/home/maldera/Desktop/eXTP/softwareXCF/XCF/libs')
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
import utils_v2 as al
import os
from scipy.optimize import curve_fit
import glob

# range nel qie cercare il picco da fittare
def fit_peak(filename, fileFormat='npz',min_range=1, max_range=100000):

   p=histogramSimo()
   p.read_from_file(filename, fileFormat )

 
   #cerco x del massimo in un certo range:
   bin_centers=fitSimo.get_centers(p.bins)
   mask=np.where((bin_centers>min_range)&(bin_centers<max_range))
   c2=p.counts[mask]
   h=np.max(c2)
   xmax=bin_centers[p.counts==h]
   print('xmax=',xmax,' h=',h)

   # qua fa il fit...
   popt,  pcov, xmin,xmax, redChi2= fitSimo.fit_Gaushistogram_iterative(p.counts,p.bins,xmin=min_range,xmax=max_range, initial_pars=[h,xmax,10], nSigma=1.5 )

   print('mean=',popt[1], ' +-',pcov[1][1]**0.5)
   print('sigma=',popt[2], ' +-',pcov[2][2]**0.5)
   print('N=',popt[0], ' +-',pcov[0][0]**0.5)
   print('CHI2/NDoF= ',redChi2)

   return  p,popt,  pcov, xmin,xmax, redChi2




XBINS=2822
YBINS=4144

commonpath='/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/8giu/'
folders= glob.glob('/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/8giu/h_*')
outdir= '/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/8giu/plots/'
nomefileout=outdir+'out.txt'
nameXdist='/X_dist_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz'
nameYdist='/Y_dist_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz'
name_image='/imageRaw_pixCut15.0sigma.fits'
name_spectrum='/spectrum_all_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz'



os.system('mkdir -p '+outdir)

meanX=[]
meanXerr=[]
meanY=[]
meanYerr=[]
title=[]

i=0
for folder in folders :


      print("folder=",folder.split('/')[-1] )
            
      x_histoname=folder+nameXdist
      y_histoname=folder+nameYdist
      imagefileName=folder+name_image
      spectName=folder+name_spectrum
      
      pX,poptX,  pcovX, xminX,xmaxX, redChi2X= fit_peak(x_histoname)
      pY,poptY,  pcovY, xminY,xmaxY, redChi2Y= fit_peak(y_histoname)
      

      meanX.append(poptX[1])
      meanY.append(poptY[1])
      meanXerr.append( pcovX[1][1]**0.5)
      meanYerr.append( pcovY[1][1]**0.5)
      title.append(folder.split('/')[-1])
      
      #draw
      fig=plt.figure(i,(18,10))
      axs = fig.subplots(2,2)
      fig.suptitle(folder.split('/')[-1])

      axs[0,0].set_title('X dist')
      pX.plot(axs[0,0],'x_dist')
      #plt.hist(pX.bins[:-1],bins=pX.bins,weights=pX.counts, histtype='step', label='x dist')
      x=np.linspace(xminX,xmaxX,1000)
      y= fitSimo.gaussian_model(x,poptX[0],poptX[1],poptX[2])
      axs[0,0].plot(x,y,'r-')
      s ='mean=' + str(round(poptX[1], 3)) + " +- " + str(round( pcovX[1][1]**0.5 ,3))+'\n sigma='+ str(round(poptX[2], 3)) + ' +- ' + str(round( pcovX[2][2]**0.5 ,3))+'\n Chi2/Ndof='+str(round(redChi2X,3)) 
      axs[0,0].text(0.7, 0.8, s,  transform = axs[0,0].transAxes,  bbox = dict(alpha = 0.7))
      axs[0,0].set_xlim([poptX[1]-4*poptX[2], poptX[1]+4*poptX[2]  ])

      # y dist:
      axs[0,1].set_title('Y dist')
      pY.plot(axs[0,1],'Y_dist')
      x=np.linspace(xminY,xmaxY,1000)
      y= fitSimo.gaussian_model(x,poptY[0],poptY[1],poptY[2])
      axs[0,1].plot(x,y,'r-')
      s = 'mean=' + str(round(poptY[1], 3)) + " +- " + str(round( pcovY[1][1]**0.5 ,3))+'\n sigma='+ str(round(poptY[2], 3)) + " +- " + str(round( pcovY[2][2]**0.5 ,3))+'\n Chi2/Ndof='+str(round(redChi2Y,3))
      axs[0,1].text(0.7, 0.8, s,  transform = axs[0,1].transAxes,  bbox = dict(alpha = 0.7))
      axs[0,1].set_xlim([poptY[1]-4*poptY[2], poptY[1]+4*poptY[2]  ])

      
      # spectrum:
      axs[1,0].set_title('counts spectrum')
      pCounts=histogramSimo()
      pCounts.read_from_file(spectName,'npz')
      pCounts.plot(axs[1,0],'counts')

      #image
      axs[1,1].set_title('counts map')
      image_data = al.read_image(imagefileName)
      axs[1,1].imshow(image_data, origin='lower',  extent=[0, XBINS, 0, YBINS])
      axs[1,1].set_ylim([poptY[1]-4*poptY[2], poptY[1]+4*poptY[2]  ])
      axs[1,1].set_xlim([poptX[1]-4*poptX[2], poptX[1]+4*poptX[2]  ])

      fig.savefig(outdir+'/'+folder.split('/')[-1]+'_all.png')
      i=i+1  

   
miofile=open(nomefileout, 'w')

for i in range (0,len(meanX)):
   print(title[i],"  x= ",meanX[i],' +- ',meanXerr[i],"y= ",meanY[i],' +- ',meanYerr[i] )
   mystring=title[i]+"  "+str(meanX[i])+' '+str(meanXerr[i])+" "+str(meanY[i])+' '+str(meanYerr[i])+'\n' 
   miofile.write(mystring)

miofile.close()   
plt.show()
