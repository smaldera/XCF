import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../libs')
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo

# range nel qie cercare il picco da fittare
min_range=1
max_range=100000

#nome file imput
filename='/home/maldera/Desktop/eXTP/data/datiSDD/misureMcPhersonGenn2023/Fe_10KV_0.1mA_orizzontale.mca'
fileFormat='sdd'

p=histogramSimo()
p.read_from_file(filename, fileFormat )

 
#cerco x del massimo in un certo range:
bin_centers=fitSimo.get_centers(p.bins)
mask=np.where((bin_centers>min_range)&(bin_centers<max_range))
c2=p.counts[mask]
h=np.max(c2)
xmax=bin_centers[p.counts==h]
print('xmax=',xmax,' h=',h)

# qua fa il fit...
popt,  pcov, xmin,xmax, redChi2= fitSimo.fit_Gaushistogram_iterative(p.counts,p.bins,xmin=min_range,xmax=max_range, initial_pars=[h,xmax,10], nSigma=1.5 )

print('mean=',popt[1], ' +-',pcov[1][1]**0.5)
print('sigma=',popt[2], ' +-',pcov[2][2]**0.5)
print('N=',popt[0], ' +-',pcov[0][0]**0.5)
print('CHI2/NDoF= ',redChi2)


# PLOTTING 
fig, ax = plt.subplots()
p.plot(ax,'test')
x=np.linspace(xmin,xmax,1000)
y= fitSimo.gaussian_model(x,popt[0],popt[1],popt[2])
plt.plot(x,y,'r-')

     
        
        
plt.show()
