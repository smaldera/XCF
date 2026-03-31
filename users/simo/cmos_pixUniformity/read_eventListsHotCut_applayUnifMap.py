import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo

from cutHotPixels import hotPixels
from mpl_toolkits.axes_grid1 import make_axes_locatable


# small scripts to plot CMOS data from the event list whit cuts
# draws: 2D map,  energy, x-y projections
#

# cut type:
# cut='x' if cut on x axis
# cut='y' if cut on y axis
# cut='xy' if cut on both axis
# cut=None does the normal w cut


def str2bool(value):
    if value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")



#cut='None'
#x_inf = 1260
#x_sup = 1700
#y_inf = 1100
#y_sup = 3175

def correct_pixGains(map_file,x_all,y_all,w_all):

    # recupero mappa:
    data=np.load("prova_unifMap.npz")
    counts2dCluAVE=data['corr']
    xedges=data['x_bins']
    yedges=data['y_bins']

    print("cerco indici")
    x_idx=np.searchsorted(xedges,x_all)-1
    y_idx=np.searchsorted(yedges,y_all)-1

    print("cerco correzioni")
    corrw= counts2dCluAVE[y_idx,x_idx]

    print("len(corrw)",len(corrw)," len(w_all[myCut])=", len(w_all))

    print("corrw=",corrw)

    return w_all/corrw


    
    



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
parser.add_argument('--hotPixelsCut',type=str2bool,  help="if true enable hot pixel cut", required=False,default=True)


    
FIND_HOTPIXELS=False
CUT_HOT_PIXELS=False
PLOT_MAP=True
map_file='prova_unifMap.npz'

args = parser.parse_args()
print("HOT PIX=",args.hotPixelsCut)
DIR = args.saveDir
ff=open(args.inFile,'r')

if (args.hotPixelsCut==False):
    FIND_HOTPIXELS=False
    CUT_HOT_PIXELS=False

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
print("Pixel Cut=",args.hotPixelsCut, "FIND_HOTPIXELS= ",FIND_HOTPIXELS," CUT_HOT_PIXELS=",CUT_HOT_PIXELS)


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])
size_all=np.array([])
timestamp_all=np.array([])

for f in ff:
    print(f[:-1])
    #w, x,y=al.retrive_vectors(f[:-1])
    #w, x,y,size=al.retrive_vectors2(f[:-1])
    #w, x,y,size,timestamp=al.retrive_vectors3(f[:-1])
    #w, x,y,size=al.retrive_vectors2(f[:-1])
    w, x,y=al.retrive_vectors(f[:-1])
   
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
  #  size_all=np.append(size_all,size)
    #timestamp_all=np.append(timestamp_all,timestamp)

print("len w_all ",len(w_all))
print("len x_all ",len(x_all))
print("len y_all ",len(y_all))

#================
#  cut hot pixels....

if FIND_HOTPIXELS==True:

    hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all,rebin=5)
    hotPix.find_HotPixels(n_sigma=10,low_threshold=50, min_counts=40) # low_treshold in ADC, 
    hotPix.save_cuts(DIR+'/cuts.npz')
if CUT_HOT_PIXELS==True:
    hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all,rebin=REBINXY)
    hotPix.retrive_cuts(DIR+'/cuts.npz')
    hotPix.applyCuts()
    w_all,   x_all,  y_all, size_all=hotPix.get_cutVectors()
                    

print("len w_all dopo cut ",len(w_all))
#===============

fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(221)

#plot
energy_all=w_all*calP1+calP0
myCut=np.where( (w_all>50))
#myCut=np.where( (w_all>50)&(energy_all<8)&(energy_all>2))
#myCut=np.where( (w_all>50)&(x_all==1000)&(y_all==1000))


#myCut=np.where( ((w_all)>50)&(x_all>5)&(x_all<2808))
#myCut=np.where( ((size_all)>1))
#myCut=np.where( (w_all>40)&( (  ((x_all-1300)**2+(y_all-1750)**2)<900**2)  ))
#myCut=np.where( (w_all>50)&(x_all>999)&(x_all<1001)&(y_all>999)&(y_all<1001) )

# mappa posizioni:
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])  \
#im=ax1.imshow((counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])  
ax1.legend()
#ax1.colorbar()
ax11=plt.subplot(222)



print("w_all[mycut]=", w_all[myCut])
print("x_all[mycut]=", x_all[myCut])
print("y_all[mycut]=", y_all[myCut])


ax2=plt.subplot(222)


w_allCorr=correct_pixGains(map_file,x_all[myCut],y_all[myCut],w_all[myCut])

# spettro energia
countsClu, binsE = np.histogram( w_all[myCut]  , bins = int(2*NBINS/2.), range = (-NBINS,NBINS) )
countsCluCorr, binsE = np.histogram( w_allCorr, bins = int(2*NBINS/2.), range = (-NBINS,NBINS) )
binsE=binsE*calP1+calP0

ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
ax2.hist(binsE[:-1], bins = binsE, weights = countsCluCorr, histtype = 'step',label="energy w. clustering -CORRECTED")

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
                
    fig4=plt.figure(figsize=(10,10))
    ax4=fig4.subplots(1,1)
    div4 = make_axes_locatable(ax4)
    cax4 = div4.append_axes('right', '5%', '5%')
    im4=ax4.imshow(counts2dClu, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])  
    #plt.colorbar()
    cb4=fig4.colorbar(im4,cax=cax4, orientation='vertical')
    ax4.set_title('counts')
   




    

if SAVE_HISTOGRAMS==True:
  
    print('... saving energy spectrun  in:', spectrum_file_name  )
    np.savez(DIR +suffix+ spectrum_file_name, counts = countsClu,  bins = binsE)
    np.savez(DIR +suffix+ xproj_file_name, counts = xprojection,  bins = bins_x)
    np.savez(DIR +suffix+ yproj_file_name, counts = yprojection,  bins = bins_y)
    fig2.savefig(DIR+suffix+'plots.png')
    
plt.show()
