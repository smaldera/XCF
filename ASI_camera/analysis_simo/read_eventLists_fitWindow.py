import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo

from scipy import optimize   
          
xf=np.empty(1)
yf=np.empty(1)
val=np.empty(1)

def myfunc(X0):
 
    
     xc=X0[0]
     yc=X0[1]
     r=X0[2]
     circle_cut=np.where((xf-xc)**2+(yf-yc)**2<r**2)
     print(circle_cut)
     return -np.sum(val[circle_cut])



fileListName='events_file_list.txt'
ff=open(fileListName,'r')


NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=20.


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])

for f in ff:
    print(f)
    w, x,y=al.retrive_vectors(f[:-1])
    print(w)
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
  

#myCut=np.where( (w_all>2390)&(w_all<2393)  )
#myCut=np.where( (x_all>800)&(x_all<1200)&(y_all>1900)&(y_all<2500)  )
myCut=np.where( (w_all>800)&(w_all<900)  )
#myCut=np.where( w_all>600 )


#plot 
counts2dClu,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu2,  xedges, yedges= np.histogram2d(x_all[myCut],y_all[myCut],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])


counts2dClu=   counts2dClu.T
plt.figure(1)
print('counts2dCluT=',counts2dClu)


#plt.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
counts2dClu[counts2dClu>0]=1
counts2dClu[counts2dClu==0]=-1


half_Xbin=(xedges[1]-xedges[0])/2.
half_Ybin=(yedges[1]-yedges[0])/2.

print("yedges[1]=",yedges[1])

i=0
# da scrivere piu' pythonic???????
for yy in range (0,ybins2d):
   for xx in range (0,xbins2d):

          #print (yy,"  bin=",yedges[yy]+half_Ybin)
          xf=np.append(xf,xedges[xx]+half_Xbin)
          yf=np.append(yf,yedges[yy]+half_Ybin)
         
          val=np.append(val,counts2dClu[yy,xx])
          i+=1
          
print("i=",i)



print(" 1 len val",len(val))
print("1 len xf",len(xf)) 
          


counts2dClu2=counts2dClu2.T
plt.imshow(np.log10(counts2dClu2), interpolation='none',    origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()


circle_fit=optimize.fmin(myfunc,[1330,2200,1000])

print("circle_fit=",circle_fit)

#plt.figure(2)
countsClu, bins = np.histogram( w_all[myCut]  , bins = 2*NBINS, range = (-NBINS,NBINS) )
#plt.hist(bins[:-1], bins = bins, weights = countsClu, histtype = 'step',label="clustering")

#myCut=np.where(((x-1000)**2+(y-1000)**2)<1000**2)
plt.plot(circle_fit[0],circle_fit[1],'ro')
c1=plt.Circle((circle_fit[0],circle_fit[1] ), radius=circle_fit[2],facecolor="none",edgecolor='red')

plt.gca().add_artist(c1)
plt.show()

