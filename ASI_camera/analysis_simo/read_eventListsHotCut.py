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
    return bin


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

calP0=-0.003201340833319255
calP1=0.003213272145961988

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=80

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
    #w, x,y=al.retrive_vectors(f[:-1])
    w, x,y,size=al.retrive_vectors2(f[:-1])
    print(w)
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
    size_all=np.append(size_all,size)



fig2=plt.figure(figsize=(10,10))
#fig2=plt.figure()
ax1=plt.subplot(221)

#plot
myCut=np.where( w_all>10 )
# mappa posizioni:
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

print('len(counts2dClu)=',len(counts2dClu), "np.shape(counts2dClu)=",np.shape(counts2dClu))




# mask hot pixels:
i_cut=[]
j_cut=[]
#for i in range(1, YBINS-1):
for i in range(1, ybins2d-1):
    
    #for j in  range(1, XBINS-1):
    for j in  range(1, xbins2d-1):
        
        counts=counts2dClu[i][j]
        
     
        mysum2=0
        for delta_i in range(-1,2):
            for delta_j in range(-1,2):
                mysum2+=counts2dClu[i+delta_i][j+delta_j]
           
        if counts<10:
            continue
        mysum2corr=(mysum2-counts)/8.
        if (counts-mysum2corr)>20.*np.sqrt(mysum2corr):
        #if counts>10:
           
            print ("AAAAGGGHHHH noise!! couts=",counts," ave =",mysum2corr ," i=",i," j=",j)
            i_cut.append(i)  #Y
            j_cut.append(j)  #X




print ("x_cut=",j_cut)
print ("y_cut=",i_cut)

for i in range(0,len(j_cut)):
    print("pix_x=",j_cut[i]," inf= ",xedges[j_cut[i]]," up=",xedges[j_cut[i]+1]  )
    print("pix_y=",i_cut[i]," inf= ",yedges[i_cut[i]]," up=",yedges[i_cut[i]+1]  )

    x_low=xedges[j_cut[i]]
    x_up=xedges[j_cut[i]+1]
    y_low=yedges[i_cut[i]]
    y_up=yedges[i_cut[i]+1]
    
    pixCut=np.where(~(((x_all<x_up)&(x_all>x_low))&( (y_all<y_up)&(y_all>y_low) )))
    w_all=w_all[pixCut]
    x_all=x_all[pixCut]
    y_all=y_all[pixCut]
    size_all= size_all[pixCut]


print(w_all)   
    
# CUT di SELEZIONE EVENTI!!!
if cut=='x':
    myCut_pos=np.where( (x_all>x_inf)&(x_all<x_sup) )
    myCut=np.where( w_all>100 )
if cut=='y':
    myCut_pos=np.where( (y_all>y_inf)&(y_all<y_sup) )
    myCut=np.where( w_all>100 )
if cut=='xy':
    myCut_pos=np.where( (x_all>x_inf)&(x_all<x_sup)&(y_all>y_inf)&(y_all<y_sup) )
    myCut=np.where( w_all>100 )
if cut=='None':
    myCut=np.where( w_all>10 )
    myCut_pos=myCut
    

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


print("inizio hitograms w")
ax2=plt.subplot(222)
# spettro energia
#plt.figure(2)
countsClu, binsE = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
binsE=binsE*calP1+calP0
ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="energy w. clustering")
ax2.set_xlabel('E[keV]')
ax2.set_xlim([0,10])
ax2.set_yscale('log')
ax2.legend()
print("... end here")


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

plt.figure(4)
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=plt.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])


if SAVE_HISTOGRAMS==True:
    DIR = args.saveDir
    print('... saving energy spectrun  in:', spectrum_file_name  )
    np.savez(DIR + spectrum_file_name, counts = countsClu,  bins = binsE)
    np.savez(DIR + xproj_file_name, counts = xprojection,  bins = bins_x)
    np.savez(DIR + yproj_file_name, counts = yprojection,  bins = bins_y)



plt.show()
