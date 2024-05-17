import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo


def cercaBin(bin_edges,val):

    bin= np.max(np.where( (bin_edges<val))[0])
 #   bin= np.max(bin_edges[bin_edges<val])
   
    return bin



XBINS=100
YBINS=200
REBINXY=1


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([1,1,1])
x_all=np.array([0,1,2])
y_all=np.array([0,0,2])


fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(111)

#plot
# mappa posizioni:
myCut=np.where(w_all>0)
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

counts2dBig,  xedgesBig, yedgesBig= np.histogram2d(x_all[myCut],y_all[myCut],bins=[int(xbins2d/2), int(ybins2d/2) ],range=[[0,XBINS],[0,YBINS]])
counts2dBig=   counts2dBig.T

fig3=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax2=plt.subplot(111)
imBig=ax2.imshow(np.log10(counts2dBig), interpolation='nearest', origin='lower',  extent=[xedgesBig[0], xedgesBig[-1], yedgesBig[0], yedgesBig[-1]])



# mask hot pixels:
for i in range(1, XBINS-1):
    for j in  range(1, YBINS-1):
        counts=counts2dClu[j][i]
        print("i=",i," j=",j," => ",counts2dClu[j][i])
        iBig=cercaBin(xedgesBig,i)
        jBig=cercaBin(yedgesBig,j)
        print("average=",counts2dBig[jBig][iBig])
        countsAve=counts2dBig[jBig][iBig]/4.
        if (counts-countsAve)>3.*np.sqrt(countsAve):
            print ("noise!! couts=",counts," ave =",countsAve," i=",i," j=",j)


#plt.colorbar()

plt.show()
