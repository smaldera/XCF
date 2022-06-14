import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT



def read_all_files(path):
    f=glob.glob(shots_path+"/shots*.npz")
    supp_weightsAll=np.empty(0)
    x_pixAll=np.empty(0)
    y_pixAll=np.empty(0)
    n=0
    for shot_file in f:
        print('===============>>>  n=',n)
        w,x_pix,y_pix=al.retrive_vectors(shot_file)
        supp_weightsAll=np.append( supp_weightsAll, w)
        x_pixAll=np.append( x_pixAll, x_pix)
        y_pixAll=np.append( y_pixAll, y_pix)
        n=n+1
    
    return supp_weightsAll,x_pixAll,y_pixAll



#####################################################

shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/source/'
bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/bkg/'
outHisto_name=shots_path+'histo_all.npz'

wAll,x_pixAll,y_pixAll= read_all_files(shots_path)

# mask:
mask=np.where( (wAll>150) & (wAll<30000)  ) # condizioni multiple con bitwise operators okkio alla precedenza!!

plt.scatter(x_pixAll[mask],y_pixAll[mask],c = np.log10(wAll[mask]) )
plt.colorbar()
plt.show()

al.retrive_histo(outHisto_name) 
counts,bins=np.histogram(wAll[mask],bins=int(65536/128.)  ,range=(0,65536/4)  )
plt.hist(bins[:-1],bins=bins,weights=counts, histtype='step')
plt.show()
