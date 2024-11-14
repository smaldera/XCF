import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

#import sys
#sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo



p=histogramSimo()
p.counts=[10,10,10,10,10,10,10,10,10,10]
p.bins=[0,1,2,3,4,5,6,7,8,9,10]

fig=plt.figure(3)
ax = fig.subplots()
p.rebin(6)

print("bins rebinnati=",p.bins)

p.plot(ax,"test!!!")

plt.show()


