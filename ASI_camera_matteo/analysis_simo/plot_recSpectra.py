import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT



#####################################################


#outHisto_name='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/Fe55/source/histo_all.npz'
outHisto_name='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/test2/CapObj/histo_all.npz'

al.retrive_histo(outHisto_name) 
#counts,bins=np.histogram(wAll[mask],bins=int(65536/128.)  ,range=(0,65536/4)  )
#plt.hist(bins[:-1],bins=bins,weights=counts, histtype='step')
plt.show()
