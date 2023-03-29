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


<<<<<<< HEAD

#filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_3/events_list_pixCut15.0sigma_CLUcut_15.0sigma.npz'
#filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-20ksx/events_list_pixCut15.0sigma_CLUcut_15.0sigma.npz'
filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/22Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx_asse2_80ksx_foro5mmLong/events_list_pixCut15.0sigma_CLUcut_15.0sigma.npz'
#filename='/home/xcf/Desktop/ASI_polarizzata/Rodio/prova_all.npz'
=======
#filename='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_orizz/Pd/100ms_G120/events_list_pixCut10.0sigma_CLUcut_5.0sigma.npz'
#filename='/home/maldera/Desktop/eXTP/data/ASI_polarizzata/Rodio/22Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx_asse2_80ksx_foro5mm/'

fileListName='events_file_list.txt'
ff=open(fileListName,'r')
>>>>>>> 0c99102693bca9d635a90f970ed2f9365c68f146

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144
<<<<<<< HEAD
REBINXY=5.
=======
REBINXY=2.
>>>>>>> 0c99102693bca9d635a90f970ed2f9365c68f146
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
  

<<<<<<< HEAD
#myCut=np.where( (w>780)&(w<860)  )
#myCut=np.where( (x>650)&(x<850)&(y>500)&(y<2500) )
#myCut=np.where( (x>250)&(x<350)&(y>500)&(y<2500) )

myCut=np.where( w>50  )
=======
#myCut=np.where( (w_all>2390)&(w_all<2393)  )
#myCut=np.where( (x_all>800)&(x_all<1200)&(y_all>1900)&(y_all<2500)  )
#myCut=np.where( (w_all>600)&(w_all<900)  )
myCut=np.where( w_all>100 )
>>>>>>> 0c99102693bca9d635a90f970ed2f9365c68f146

#plot 
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
plt.figure(1)
<<<<<<< HEAD
plt.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
=======
plt.imshow(np.log10(counts2dClu), interpolation='none', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
>>>>>>> 0c99102693bca9d635a90f970ed2f9365c68f146
plt.colorbar()

plt.figure(2)
countsClu, bins = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="clustering")

#plt.figure(3)
#plt.plot(x[myCut],y[myCut],'or',alpha=0.01)


plt.show()
