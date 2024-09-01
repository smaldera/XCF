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


def read_allCMOS(cmos_eventsFiles,binsSdd):


    # retta calibrazione cmos
    calP0=-0.003201340833319255
    calP1=0.003213272145961988
 
    counts=[]
    counts_all=0.
    #livetime_all=0
    bins=0
 
    for i in range(0,len(  cmos_eventsFiles)):

        f=common_path+'cmos/efficency_test/'+cmos_eventsFiles[i]
        print("reading: ",f)
        w, x,y,size=al.retrive_vectors2(f)
        print("len w =",w)
        energies=w*calP1+calP0
        #taglio spaziale!!!! 
        #mask=np.where( (x-xc)**2+(y-yc)**2<r )
        countsClu, binsE = np.histogram( energies  , bins =len(binsSdd)-1, range = (binsSdd[0],binsSdd[-1]) )
        plt.hist(binsE[:-1],bins=binsE ,weights=countsClu, histtype='step', label="cmos")
        plt.legend()
        if i==0:
           counts_all=countsClu
        else:
            counts_all+=countsClu 
        
    plt.title('cmos')

    return binsE,counts_all 

    

if __name__ == "__main__":

    common_path='/home/maldera/Desktop/eXTP/data/CMOS_efficiency/'
  
    #dati sdd"
    mca_file=['UnPol_10kV_0.01mA_set1.mca','UnPol_10kV_0.01mA_set2.mca','UnPol_10kV_0.01mA_set3.mca','UnPol_10kV_0.01mA_set4.mca']

    plt.figure(1)
    binsSdd,counts_all,livetime_all=read_allSdd(common_path, mca_file)
               
    #plot SDD
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.title('SDD')
    plt.hist(binsSdd[:-1],bins=binsSdd ,weights=counts_all/livetime_all, histtype='step', label="spettro somma")
    plt.legend()  
    plt.title('SDD')

    ### read CMOS data:
    list_name='events_list_pixCut10sigma_CLUcut_10sigma_v2.npz'
    cmos_eventsFiles=['/DATA1/'+list_name, '/DATA2/'+list_name, '/DATA3/'+list_name, '/DATA4/'+list_name]
    plt.figure(2)
    binsCmos,counts_allCmos= read_allCMOS(cmos_eventsFiles,binsSdd)
    plt.hist(binsCmos[:-1],bins=binsCmos ,weights=counts_allCmos, histtype='step', label="spettro somma")
    plt.legend()  


    print("binsSdd=",binsSdd)
    print("binsCmos=",binsCmos)
  

    #rebinno gli istogrammi??
    psdd=histogramSimo()
    psdd.counts=counts_all
    psdd.bins=binsSdd

    pcmos=histogramSimo()
    pcmos.counts=counts_allCmos
    pcmos.bins=binsCmos

    pcmos.rebin(100)
    psdd.rebin(100)

    fig=plt.figure(3)
    ax = fig.subplots()
    pcmos.plot(ax,"cmos")
    psdd.plot(ax,"sdd")
    plt.legend()
   
    plt.show()


    


