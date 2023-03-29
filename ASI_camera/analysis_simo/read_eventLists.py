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



#filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_3/events_list_pixCut15.0sigma_CLUcut_15.0sigma.npz'
#filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-20ksx/events_list_pixCut15.0sigma_CLUcut_15.0sigma.npz'
filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/22Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx_asse2_80ksx_foro5mmLong/events_list_pixCut15.0sigma_CLUcut_15.0sigma.npz'
#filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/prova_all.npz'

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144
REBINXY=5.
xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w, x,y=al.retrive_vectors(filename)

#myCut=np.where( (w>780)&(w<860)  )
#myCut=np.where( (x>650)&(x<850)&(y>500)&(y<2500) )
#myCut=np.where( (x>250)&(x<350)&(y>500)&(y<2500) )

myCut=np.where( w>50  )

#plot 
counts2dClu,  xedges, yedges= np.histogram2d(x[myCut],y[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
plt.figure(1)
plt.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()

plt.figure(2)
countsClu, bins = np.histogram( w[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="clustering")

#plt.figure(3)
#plt.plot(x[myCut],y[myCut],'or',alpha=0.01)


plt.show()
