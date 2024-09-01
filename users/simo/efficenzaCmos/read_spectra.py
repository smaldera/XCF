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


mpl.rcParams["font.size"] = 15

def read_allSdd(common_path, mca_file):
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    
    #common_path='/home/maldera/Desktop/eXTP/data/CMOS_efficiency/sdd/'
  
    #"dati sdd"
    #mca_file=['UnPol_10kV_0.01mA_set1.mca','UnPol_10kV_0.01mA_set2.mca','UnPol_10kV_0.01mA_set3.mca','UnPol_10kV_0.01mA_set4.mca']
    

    livetime=[]
    counts=[]
    counts_all=0.
    livetime_all=0
    bins=0
    
    for i in range (0,len(mca_file)):

       p=histogramSimo()
       filename=common_path+'sdd/'+mca_file[i]            
       p.read_from_file(filename, 'sdd' )
       print("livetime=",p.sdd_liveTime,"counts=", p.sdd_fastCounts, "RATE=",p.sdd_fastCounts/p.sdd_liveTime,' Hz' )
       print("deadTime=",p.sdd_deadTime)
       
       # calibrazione energia
       p.bins=p.bins*calP1+calP0

       
       mylabel=mca_file[i][0:-4]
       plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts/p.sdd_liveTime, histtype='step', label=mylabel)
       livetime.append(p.sdd_liveTime)
       counts.append(p.counts)
    
       
       #sum livetimes nad counts:
       if i==0:
           counts_all=p.counts
       else:
            counts_all+=p.counts
       livetime_all+=p.sdd_liveTime
       bins=p.bins

        
    #end loop
    return bins,counts_all,livetime_all





if __name__ == "__main__":

    common_path='/home/maldera/Desktop/eXTP/data/CMOS_efficiency/'
  
    #dati sdd"
    mca_file=['UnPol_10kV_0.01mA_set1.mca','UnPol_10kV_0.01mA_set2.mca','UnPol_10kV_0.01mA_set3.mca','UnPol_10kV_0.01mA_set4.mca']

    plt.figure(1)
    bins,counts_all,livetime_all=read_allSdd(common_path, mca_file)
               
    #plot SDD
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.title('SDD')
    plt.hist(bins[:-1],bins=bins ,weights=counts_all/livetime_all, histtype='step', label="spettro somma")
    plt.legend()  
    plt.title('SDD')

    




    
    plt.show()


    


