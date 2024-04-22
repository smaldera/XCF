import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
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

x_inf = 0
x_sup = 2000
y_inf = 1000
y_sup = 2000


import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-in','--inFile', type=str,  help='txt file with list of npz files', required=True)
parser.add_argument('-dir','--saveDir', type=str,  help='direxctory where npz files are saved', required=False)

args = parser.parse_args()




ff=open(args.inFile,'r')


# retta calibrazione cmos
#calP1=0.0015013787118821926
#calP0=-0.03544731540487446
calP1= 0.00321327
calP0=-0.0032013

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=40.

SAVE_HISTOGRAMS=True
spectrum_file_name='test_spectrum.npz'
xproj_file_name='test_xproj.npz'
yproj_file_name='test_yproj.npz'


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])
y_all=np.array([])
size_all=np.array([])


for f in ff:
    print(f)
   # w, x,y,size=al.retrive_vectors(f[:-1])
    w, x,y=al.retrive_vectors_old(f[:-1])
   
   # print(size)
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
    #size_all=np.append(size_all,size)


#mask_n=[1]*1e6
    
print ("size_all=",size_all)

# CUT di SELEZIONE EVENTI!!!
if cut=='x':
    myCut_pos=np.where( (x_all>x_inf)&(x_all<x_sup) )
    myCut=np.where( w_all>100 )
if cut=='y':
    myCut_pos=np.where( (y_all>y_inf)&(y_all<y_sup) )
    myCut=np.where( w_all>100 )
if cut=='xy':
    myCut_pos=np.where( (x_all>x_inf)&(x_all<x_sup)&(y_all>y_inf)&(y_all<y_sup) )
    myCut=np.where( w_all>0 )
if cut=='None':
    myCut=np.where(( w_all>40))
   # myCut=np.where(( ( w_all*calP1+calP0)<10.5 )&( ( w_all*calP1+calP0)>9.5)   )
    #myCut=np.where(  ((w_all*calP1+calP0)>0.03)& (size==2) )
    #myCut=np.where( (x_all>x_inf)&(x_all<x_sup)&(y_all>y_inf)&(y_all<y_sup)& (w_all>100)  ) 
    myCut_pos=myCut
    
#myCut=np.where( (w_all>2390)&(w_all<2393)  )
# myCut=np.where( (x_all>800)&(x_all<1200)&(y_all>1900)&(y_all<2500)  )
# myCut=np.where( (x_all>1950)&(x_all<2420))

#myCut=np.where( (w_all>800)&(w_all<900)  )


print("n eventi=",len(w_all))


fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(221)

#plot 
# mappa posizioni:
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut][0:1000000],y_all[myCut][0:1000000],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
if cut=='x':
    ax1.axvline(x=x_inf,color='red',linestyle='--')
    ax1.axvline(x=x_sup,color='red',linestyle='--')
if cut=='y':
    ax1.axhline(y=y_inf,color='red',linestyle='--')
    ax1.axhline(y=y_sup,color='red',linestyle='--')
if cut=='xy':
    ax1.axvline(x=x_inf,color='red',linestyle='--')
    ax1.axvline(x=x_sup,color='red',linestyle='--')
    ax1.axhline(y=y_inf,color='red',linestyle='--')
    ax1.axhline(y=y_sup,color='red',linestyle='--')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im,ax=ax1)
ax1.legend()

ax2=plt.subplot(222)
# spettro energia
#plt.figure(2)
countsClu, binsE = np.histogram( w_all[myCut_pos]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
binsE=binsE*calP1+calP0

mySelection=np.where( ((w_all*calP1+calP0)>5.7)& ((w_all*calP1+calP0)<6.7)    )
print("n eventi!!! =",len(w_all[mySelection]))



print()
print("AAAAAAAAAAAAAAAAAAAAAAAA")
print("y = ", countsClu)
print("x = ", binsE)
print()
ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
ax2.set_xlabel('E[keV]')
ax2.set_xlim([0,15])
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
