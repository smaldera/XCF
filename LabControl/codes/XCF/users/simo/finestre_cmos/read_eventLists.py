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

# cut type:
# cut='x' if cut on x axis
# cut='y' if cut on y axis
# cut='xy' if cut on both axis
# cut=None does the normal w cut

cut='None'

x_inf = 1260
x_sup = 1700
y_inf = 1100
y_sup = 3175


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

NBINS=int(16384/50.)  # n.canali ADC (2^14)
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

myCut=np.where( (w_all>100)&(y_all>1000)&(y_all<3500)&(x_all>250))



fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(221)

#plot 
# mappa posizioni:
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='upper',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im,ax=ax1)


ax2=plt.subplot(222)
# spettro energia
#plt.figure(2)
countsClu, binsE = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-16384,16384) )
binsE=binsE*calP1+calP0
ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
ax2.set_xlabel('E[keV]')
ax2.set_xlim([0,10])
ax2.set_yscale('log')
ax2.legend()


# proiezione X:
#plt.figure(3)
ax3=plt.subplot(223)
xprojection, bins_x = np.histogram( x_all[myCut]  , bins =xbins2d , range = (0,XBINS) )
ax3.hist(bins_x[:-1], bins = bins_x, weights = xprojection, histtype = 'step',label="x-projection")
if cut=='x':
    ax3.axvline(x=x_inf,color='red',linestyle='--')
    ax3.axvline(x=x_sup,color='red',linestyle='--')
if cut=='xy':
    ax3.axvline(x=x_inf,color='red',linestyle='--')
    ax3.axvline(x=x_sup,color='red',linestyle='--')
ax3.legend()
ax3.set_yscale('log')

# proiezione y:
#plt.figure(4)
ax4=plt.subplot(224)
yprojection, bins_y = np.histogram( y_all[myCut]  , bins =ybins2d , range = (0,YBINS) )
ax4.hist(bins_y[:-1], bins = bins_y, weights = yprojection, histtype = 'step',label="y-projection")
if cut=='y':
    ax4.axvline(x=y_inf,color='red',linestyle='--')
    ax4.axvline(x=y_sup,color='red',linestyle='--')
if cut=='xy':
    ax4.axvline(x=y_inf,color='red',linestyle='--')
    ax4.axvline(x=y_sup,color='red',linestyle='--')
ax4.legend()
ax4.set_yscale('log')


print("w_all[myCut].size()=",w_all.size )


if SAVE_HISTOGRAMS==True:
    DIR = args.saveDir
    print('... saving energy spectrun  in:', spectrum_file_name  )
    np.savez(DIR + spectrum_file_name, counts = countsClu,  bins = binsE)
    np.savez(DIR + xproj_file_name, counts = xprojection,  bins = bins_x)
    np.savez(DIR + yproj_file_name, counts = yprojection,  bins = bins_y)
    


plt.show()
