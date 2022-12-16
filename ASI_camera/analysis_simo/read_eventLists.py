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


#filename='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_orizz/Pd/100ms_G120/events_list_pixCut10.0sigma_CLUcut_5.0sigma.npz'
filename='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_verticale/Pd/1ms_G120/events_list_pixCut10.0sigma_CLUcut_5.0sigma.npz'

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144
REBINXY=20.
xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w, x,y=al.retrive_vectors(filename)

myCut=np.where( (w>2390)&(w<2393)  )
#myCut=np.where( (x>800)&(x<1200)&(y>1900)&(y<2500)  )
#myCut=np.where( w>0  )

#plot 
counts2dClu,  xedges, yedges= np.histogram2d(x[myCut],y[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
plt.figure(1)
plt.imshow(counts2dClu, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()

plt.figure(2)
countsClu, bins = np.histogram( w[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="clustering")


plt.show()
