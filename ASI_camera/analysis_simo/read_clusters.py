import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import utils_v2 as al


XBINS=2822
YBINS=4144
NBINS=16384  # n.canali ADC (2^14)
calP0=-0.003201340833319255
calP1=0.003213272145961988



files=glob.glob('/home/maldera/Desktop/eXTP/data/2/img_cluSize10*.npz')
print(files)

for myfile in files:
    loaded=np.load(myfile)
    x=loaded['x']
    y=loaded['y']
    w=loaded['w']

    w=w*calP1+calP0

    # creo histo2d:
    countsCharge,  xedges, yedges=       np.histogram2d(x,y,weights=w,bins=[XBINS, YBINS],range=[[0,XBINS],[0,YBINS]])
    countsCharge=  countsCharge.T
    plt.imshow(countsCharge, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    plt.xlim(min(x)-10,max(x)+10 )
    plt.ylim(min(y)-10,max(y)+10) 
    plt.show()


                 
