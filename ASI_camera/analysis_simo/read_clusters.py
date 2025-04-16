import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import utils_v2 as al

from scipy.stats import pearsonr

import fit_histogram as fh
import pylandau
  
XBINS=2822
YBINS=4144
NBINS=16384  # n.canali ADC (2^14)
calP0=-0.003201340833319255
calP1=0.003213272145961988

PLOT=False
n_energyBins=2000
#16384

files=glob.glob('/home/maldera/Desktop/eXTP/data/CMOS_verticale/clusters/tracks*/*img*.npz')


print('n immagini=',len(files))

x = []
countsE_all, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))
countsE, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))

countsETrack_all, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))
countsETrack, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))


#print(files)

r_all=[]
size_all=[]
totE=[]

print("n tracks=",len(files))

for myfile in files:
   # print ("reading file: ",myfile)
    loaded=np.load(myfile)
    x=loaded['x']
    y=loaded['y']
    w=loaded['w']

    w=w*calP1+calP0
    corr, _ = pearsonr(x, y)
    r_all.append(abs(corr))
    #print ("corr=",corr)
    size_all.append(len(w))
    totE.append(np.sum(w))

    w_cut=np.where(w>0.2)
    
    w=w[w_cut]
    x=x[w_cut]
    y=y[w_cut]

    if len(w)==7 and abs(corr)>0.6:
        countsETrack, binsE = np.histogram(np.sum(w), bins =n_energyBins, range = (0,100))
        countsETrack_all=countsETrack_all+countsETrack
        
       # print( np.sum(w[0:2]),"  ",np.sum(w[-2:]))
        if np.sum(w[0:2])/2.>np.sum(w[-2:])/2. and w[0]>0:
            countsE, binsE = np.histogram(w[0], bins =n_energyBins, range = (0,100))
        if np.sum(w[0:2])/2.<=np.sum(w[-2:])/2. and w[-1]>0:
           countsE, binsE = np.histogram(w[-1], bins =n_energyBins, range = (0,100))
           countsE_all=countsE_all+countsE

       
        
        #for wi in w:
        #    if wi>30:
        #        print("myfile=",myfile)
        
   # if PLOT==True and abs(corr)<0.001:
    if PLOT==True and abs(corr)>0.9 and len(w)==15:


        print("x=",x)
        print("y=",y)
        print("w=",w)
        
        
        # creo histo2d:
        countsCharge,  xedges, yedges=       np.histogram2d(x,y,weights=w,bins=[XBINS, YBINS],range=[[0,XBINS],[0,YBINS]])
        countsCharge=  countsCharge.T
        plt.imshow(countsCharge, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar()
        plt.xlim(min(x)-10,max(x)+10 )
        plt.ylim(min(y)-10,max(y)+10) 
        plt.show()


#plot E histogram                                  
plt.figure(2)
plt.hist(binsE[:-1], bins =binsE, weights =countsE_all  , histtype = 'step',label='Etracks')
#plt.legend()
plt.title('E traks')
# fit landau... 

coeff,pcov= fh.fit_Langau_histogram(countsE_all,binsE,xmin=-1,xmax=3.5, initial_pars=[0.5,0.3,  0.1, 16.01601151], parsBoundsLow=-np.inf, parsBoundsUp=np.inf )
#coeff,pcov= fh.fit_Landau_histogram(countsE_all,binsE,xmin=-1,xmax=3.5, initial_pars=[0.4,0.1,10], parsBoundsLow=-np.inf, parsBoundsUp=np.inf )

print ("coeff=",coeff)
x=np.arange(-1,3.5,0.05)
plt.plot(x, pylandau.langau(x, *coeff), "-")
#plt.plot(x, pylandau.langau(x, 0.5,0.3,  0.0047145326, 16.01601151), "-")
#plt.plot(x, pylandau.landau(x, *coeff), "-")

############


plt.figure(3)
counts_r,bins_r= np.histogram(r_all, bins =100, range = (-1,1))
plt.hist(bins_r[:-1], bins =bins_r, weights=counts_r  , histtype = 'step')
#plt.legend()
plt.title('track r')

plt.figure(4)
counts_s,bins_s= np.histogram(size_all, bins =100, range = (0,100))
plt.hist(bins_s[:-1], bins =bins_s, weights=counts_s  , histtype = 'step')
#plt.legend()
plt.title('track size')


# plot r vs energy
plt.figure(5) 
plt.plot(totE,r_all,"p", alpha=0.5)
plt.xlabel("cluster E [keV]")
plt.ylabel("abs(cluster lienar correlation coef.)")

# plot r vs energy
plt.figure(6)

plt.ylabel("abs(cluster lienar correlation coef.)")
plt.hist(binsE[:-1], bins =binsE, weights=countsETrack_all, histtype = 'step')
#plt.legend()
plt.title('track totalE')


plt.show()
                                  
                                  

                                  
                                  
                 
