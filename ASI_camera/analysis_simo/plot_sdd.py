import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca








if __name__ == "__main__":


    common_path='/home/maldera/Desktop/eXTP/datiSDD/mcPhersonNov2022/'
    mca_file=['Fe_10KV_0.0mA.mca','Fe_10KV_0.1mA.mca','Fe_8KV_0mA.mca','Mo_4KV_0.1mA.mca','Ti_6KV_0.0mA.mca','Ti_7KV_0.0mA.mca','Ti_8KV_0.0mA.mca']

    fig1=plt.figure(1)
    
    for i in range (0,len(mca_file)):
       data_array, deadTime, livetime, fast_counts =pharse_mca(common_path+mca_file[i])
       print("livetime=",livetime,"counts=", fast_counts, "RATE=",fast_counts/livetime,' Hz' )
       print("deadTime=",deadTime)
       size=len(data_array)      
       bin_edges=np.linspace(0,size+1,size+1)
       mylabel=mca_file[i][0:-4]
       plt.hist(bin_edges[:-1],bins=bin_edges,weights=data_array, histtype='step', label=mylabel)


    #plot
    plt.xlabel('MCA channels')
    plt.ylabel('counts')
    plt.legend()  
    plt.title('SDD -  collimatore 2mm')
    plt.show()




