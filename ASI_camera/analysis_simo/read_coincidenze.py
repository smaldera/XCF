import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
from cutHotPixels import hotPixels
from tqdm import  tqdm


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

parser.add_argument('-dir','--saveDir', type=str,  help='direxctory where npz files are saved', required=False)
parser.add_argument('-calP0','--calP0', type=float,  help='0 cal parameter', required=False,default=-0.003201340833319255)
parser.add_argument('-calP1','--calP1', type=float,  help='1 cal parameter', required=False,default=0.003213272145961988)
parser.add_argument('-rebinxy','--rebinxy', type=int,  help='x-y rebin', required=False,default=20)
parser.add_argument('-nosave','--nosaveHistos', action='store_false',  help="don't save histograms", required=False)
parser.add_argument('-specName','--specName',type=str ,  help="spectrum file name", required=False,default='test_spectrum.npz')
parser.add_argument('-xprojName','--xprojName',type=str ,  help="x-projection file name", required=False,default='test_xproj.npz')
parser.add_argument('-yprojName','--yprojName',type=str ,  help="y-projection file name", required=False,default='test_yproj.npz')
parser.add_argument('-suffix','--suffix',type=str ,  help="suffix in file names", required=False,default='coincidenza')


FIND_HOTPIXELS=True
CUT_HOT_PIXELS=True
PLOT_MAP=True

args = parser.parse_args()
#DIR = args.saveDir
DIR='/home/maldera/Desktop/eXTP/data/testCMOS_coincidenze/29Jul25/'


files_0='/home/maldera/Desktop/eXTP/data/testCMOS_coincidenze/29Jul25/camera0/file_list.txt'
files_1='/home/maldera/Desktop/eXTP/data/testCMOS_coincidenze/29Jul25/camera1/file_list.txt'
ff_0=open(files_0,'r')

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
time_all=np.array([])

for f in ff_0:
    print(f)
    #w, x,y=al.retrive_vectors(f[:-1])
    w, x,y,size, time=al.retrive_vectors3(f[:-1])
    print("LENs w=",len(w)," x=",len(x)," time=",len(time))
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
    size_all=np.append(size_all,size)
    time_all=np.append(time_all,time)
 
    
print("len w_all ",len(w_all))
print("len x_all ",len(x_all))
print("len y_all ",len(y_all))
print("len time_all ",len(time_all))

# applico hotPixelscut... 
hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all,time_all=time_all,rebin=10)
hotPix.find_HotPixels(n_sigma=4,low_threshold=60, min_counts=10) # low_treshold in ADC, 
hotPix.applyCuts()
w_all,   x_all,  y_all, size_all,time_all=hotPix.get_cutVectors()

print("len w_all  after hotpix cut:",len(w_all))


w1_all=np.array([])
x1_all=np.array([])
y1_all=np.array([])
size1_all=np.array([])
time1_all=np.array([])


############################3
ff_1=open(files_1,'r')

for f in ff_1:
    print(f)
    #w, x,y=al.retrive_vectors(f[:-1])
    w, x,y,size, time=al.retrive_vectors3(f[:-1])  
    w1_all=np.append(w1_all,w)
    x1_all=np.append(x1_all,x)
    y1_all=np.append(y1_all,y)
    size1_all=np.append(size1_all,size)
    time1_all=np.append(time1_all,time)
 
    
print("len w1_all ",len(w1_all))
print("len x1_all ",len(x1_all))
print("len y1_all ",len(y1_all))
print("len time1_all ",len(time1_all))

# applico hotPixelscut... 
hotPix=hotPixels(x_all=x1_all,y_all=y1_all,w_all=w1_all,size_all=size1_all,time_all=time1_all,rebin=20)
hotPix.find_HotPixels(n_sigma=3,low_threshold=60, min_counts=10) # low_treshold in ADC, 
hotPix.applyCuts()
w1_all,   x1_all,  y1_all, size1_all,time1_all=hotPix.get_cutVectors()

print("len w1_all  after hotpix cut:",len(w1_all))


myCut=np.where( (w_all>55)&(x_all>5)&(x_all<2808) )
myCut1=np.where( (w1_all>55)&(x1_all>5)&(x1_all<2808) )




# applico  cut sulle singole:
w1_all=w1_all[myCut1]
x1_all=x1_all[myCut1]
y1_all=y1_all[myCut1]
size1_all=size1_all[myCut1]
time1_all=time1_all[myCut1]

w_all=w_all[myCut]
x_all=x_all[myCut]
y_all=y_all[myCut]
size_all=size_all[myCut]
time_all=time_all[myCut]
print("len w1_all  after w1_all>80:",len(w1_all))
print("len w_all  after w_all>80:",len(w_all))


w1cc=[]
w0cc=[]




##### cerco coincidenze:

for i in tqdm(range(0,len(w_all))):
    cut_1=np.where( np.abs(time1_all-time_all[i])<0.250 )[0]
    #if len(cut_1)>0:
    if len(cut_1)==1:
      
       w1cc.append(w1_all[cut_1][0])
       w0cc.append(w_all[i])
      # print ("trovata coinc  len=",len(cut_1))

    if len(w0cc)>500000:
        print("break!!!!")
        break

    
fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(221)

#plot


# mappa posizioni:
counts2dClu,  xedges, yedges= np.histogram2d(x_all,y_all,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax1.legend()
ax2=plt.subplot(222)

# spettro energia
#countsClu, binsE = np.histogram( w_all[myCut]  , bins = int(2*NBINS/2.), range = (-NBINS,NBINS) )
#countsClu1, binsE = np.histogram( w1_all[myCut1]  , bins = int(2*NBINS/2.), range = (-NBINS,NBINS) )

countsClu, binsE = np.histogram( np.array(w0cc)  , bins = int(2*NBINS/5), range = (-NBINS,NBINS) )
countsClu1, binsE = np.histogram( np.array(w1cc)  , bins = int(2*NBINS/5), range = (-NBINS,NBINS) )

binsE=binsE*calP1+calP0
ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering cmos 0")
ax2.set_xlabel('E[keV]')
#ax2.set_xlim([0,12])
ax2.set_yscale('log')
ax2.legend()



#plt.figure(3)
ax3=plt.subplot(223)
counts2dClu1,  xedges, yedges= np.histogram2d(x1_all,y1_all,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu1=   counts2dClu1.T
im=ax3.imshow(np.log10(counts2dClu1), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax3.legend()


ax4=plt.subplot(224)
ax4.hist(binsE[:-1], bins = binsE, weights = countsClu1, histtype = 'step',label="energy w. clustering cmos 1")
ax4.set_xlabel('E[keV]')
#ax2.set_xlim([0,12])
ax4.set_yscale('log')
ax4.legend()

#xprojection, bins_x = np.histogram( x_all[myCut]  , bins =xbins2d , range = (0,XBINS) )
#ax3.hist(bins_x[:-1], bins = bins_x, weights = xprojection, histtype = 'step',label="x-projection")
#ax3.legend()
#ax3.set_yscale('log')

# proiezione y:
#plt.figure(4)
#ax4=plt.subplot(224)
#yprojection, bins_y = np.histogram( y_all[myCut]  , bins =ybins2d , range = (0,YBINS) )
#ax4.hist(bins_y[:-1], bins = bins_y, weights = yprojection, histtype = 'step',label="y-projection")
#ax4.legend()
#ax4.set_yscale('log')




if SAVE_HISTOGRAMS==True:
  
    print('... saving energy spectrun  in:', spectrum_file_name  )
    np.savez(DIR +suffix+ spectrum_file_name, counts = countsClu,  bins = binsE)
    np.savez(DIR +suffix+'comos1_'+ spectrum_file_name, counts = countsClu1,  bins = binsE)
    
    #np.savez(DIR +suffix+ xproj_file_name, counts = xprojection,  bins = bins_x)
    #np.savez(DIR +suffix+ yproj_file_name, counts = yprojection,  bins = bins_y)
    fig2.savefig(DIR+suffix+'plots.png')
    
plt.show()
