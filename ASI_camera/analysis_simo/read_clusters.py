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



#files=glob.glob('/home/maldera/Desktop/eXTP/data/2/img_cluSize10*.npz')
files=glob.glob('/home/maldera/Desktop/eXTP/data/CMOS_verticale/tracks*/*img*.npz')
#files=glob.glob('/home/maldera/Desktop/eXTP/data/CMOS_verticale/tracks/img_cluSize10_4624_419_4.npz')


x = []
countsE_all, binsE = np.histogram(x, bins =5000, range = (0,100))

#print(files)

for myfile in files:
   # print ("reading file: ",myfile)
    loaded=np.load(myfile)
    x=loaded['x']
    y=loaded['y']
    w=loaded['w']

    w=w*calP1+calP0
    if len(w)>0:
        countsE, binsE = np.histogram(w, bins =5000, range = (0,100))
        countsE_all=countsE_all+countsE
        for wi in w:
            if wi>30:
                print("myfile=",myfile)
        
    
    # creo histo2d:
    #countsCharge,  xedges, yedges=       np.histogram2d(x,y,weights=w,bins=[XBINS, YBINS],range=[[0,XBINS],[0,YBINS]])
    #countsCharge=  countsCharge.T
    #plt.imshow(countsCharge, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #plt.colorbar()
    #plt.xlim(min(x)-10,max(x)+10 )
    #plt.ylim(min(y)-10,max(y)+10) 
    #plt.show()


#plot E histogram                                  
plt.figure(2)
plt.hist(binsE[:-1], bins =binsE, weights =countsE_all  , histtype = 'step',label='Etracks')
#plt.legend()
plt.title('E traks')

plt.show()
                                  
                                  

                                  
                                  
                 
