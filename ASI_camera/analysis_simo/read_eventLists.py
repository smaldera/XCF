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
ff=open(fileListName,'r')


NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=30.


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

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
myCut=np.where( (w_all>800)&(w_all<900)  )
#myCut=np.where( w_all>600 )


#plot 
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
plt.figure(1)


plt.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#counts2dClu[counts2dClu>0]=1
#plt.imshow(counts2dClu, interpolation='none',    origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

plt.colorbar()

plt.figure(2)
countsClu, bins = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="clustering")


plt.show()
