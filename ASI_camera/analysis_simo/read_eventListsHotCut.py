import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo

from cutHotPixels import hotPixels



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
parser.add_argument('-calP0','--calP0', type=float,  help='0 cal parameter', required=False,default=-0.003201340833319255)
parser.add_argument('-calP1','--calP1', type=float,  help='1 cal parameter', required=False,default=0.003213272145961988)
parser.add_argument('-rebinxy','--rebinxy', type=int,  help='x-y rebin', required=False,default=20)
parser.add_argument('-nosave','--nosaveHistos', action='store_false',  help="don't save histograms", required=False)
parser.add_argument('-specName','--specName',type=str ,  help="spectrum file name", required=False,default='test_spectrum.npz')
parser.add_argument('-xprojName','--xprojName',type=str ,  help="x-projection file name", required=False,default='test_xproj.npz')
parser.add_argument('-yprojName','--yprojName',type=str ,  help="y-projection file name", required=False,default='test_yproj.npz')
parser.add_argument('-suffix','--suffix',type=str ,  help="suffix in file names", required=False,default='')


FIND_HOTPIXELS=False
CUT_HOT_PIXELS=False
PLOT_MAP=True

args = parser.parse_args()
DIR = args.saveDir
ff=open(args.inFile,'r')

# retta calibrazione cmos
calP0=args.calP0
calP1=args.calP1

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=args.rebinxy
SAVE_HISTOGRAMS=args.nosaveHistos
spectrum_file_name=args.specName 
xproj_file_name=args.xprojName  
yproj_file_name=args.yprojName
suffix=args.suffix

print("calP0=",calP0,"  calP1=",calP1)
print("Rebin xy=",REBINXY)
print("Save histograms=",SAVE_HISTOGRAMS)
print("spectrum name=",spectrum_file_name)
print("x projection name=",xproj_file_name)
print("y projection name=",yproj_file_name)
print("suffix=",suffix)


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])
size_all=np.array([])

for f in ff:
    print(f)
    #w, x,y=al.retrive_vectors(f[:-1])
    w, x,y,size=al.retrive_vectors2(f[:-1])  
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
    size_all=np.append(size_all,size)

print("len w_all ",len(w_all))
print("len x_all ",len(x_all))
print("len y_all ",len(y_all))

#================
#  cut hot pixels....


if FIND_HOTPIXELS==True:
    hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all,rebin=20)
    hotPix.find_HotPixels(n_sigma=5,low_threshold=10, min_counts=10) # low_treshold in ADC, 
    hotPix.save_cuts(DIR+'/cuts.npz')
if CUT_HOT_PIXELS==True:
    hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all)
    hotPix.retrive_cuts(DIR+'/cuts.npz')
    hotPix.applyCuts()
    w_all,   x_all,  y_all, size_all=hotPix.get_cutVectors()
                    

print("len w_all dopo cut ",len(w_all))
#===============

fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(221)

#plot
#myCut=np.where( ((w_all)>50))
#myCut=np.where( ((size_all)>1))
#myCut=np.where( (w_all>40)&( (  ((x_all-1300)**2+(y_all-1750)**2)<900**2)  ))
myCut=np.where( (w_all>40)&(x_all>1060)&(x_all<1980)&(y_all>1475)&(y_all<2660) )




# mappa posizioni:
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])  
ax1.legend()
ax2=plt.subplot(222)

# spettro energia
countsClu, binsE = np.histogram( w_all[myCut]  , bins = int(2*NBINS/20.), range = (-NBINS,NBINS) )
binsE=binsE*calP1+calP0
ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
ax2.set_xlabel('E[keV]')
#ax2.set_xlim([0,12])
ax2.set_yscale('log')
ax2.legend()



# proiezione X:
#plt.figure(3)
ax3=plt.subplot(223)
xprojection, bins_x = np.histogram( x_all[myCut]  , bins =xbins2d , range = (0,XBINS) )
ax3.hist(bins_x[:-1], bins = bins_x, weights = xprojection, histtype = 'step',label="x-projection")
ax3.legend()
ax3.set_yscale('log')

# proiezione y:
#plt.figure(4)
ax4=plt.subplot(224)
yprojection, bins_y = np.histogram( y_all[myCut]  , bins =ybins2d , range = (0,YBINS) )
ax4.hist(bins_y[:-1], bins = bins_y, weights = yprojection, histtype = 'step',label="y-projection")
ax4.legend()
ax4.set_yscale('log')


print("w_all[myCut].size()=",w_all.size )


if PLOT_MAP==True:
    fig3=plt.figure(figsize=(10,10))
    plt.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])  
    #plt.legend()
    plt.colorbar()
    plt.title('log10(counts)')


if SAVE_HISTOGRAMS==True:
  
    print('... saving energy spectrun  in:', spectrum_file_name  )
    np.savez(DIR +suffix+ spectrum_file_name, counts = countsClu,  bins = binsE)
    np.savez(DIR +suffix+ xproj_file_name, counts = xprojection,  bins = bins_x)
    np.savez(DIR +suffix+ yproj_file_name, counts = yprojection,  bins = bins_y)
    fig2.savefig(DIR+suffix+'plots.png')
    
plt.show()
