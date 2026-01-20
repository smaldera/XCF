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



def compute_Ratios(c1,c2,s1,s2):
       r=c1/c2
       rErr=np.sqrt( (s1**2)*((1/c2)**2)+(s2**2)*(c1/(c2**2))**2)
       return r, rErr
       
def compute_HistRatios(p1,p2):
    x=fitSimo.get_centers(p1.bins)
    p1=p1.counts
    p2=p2.counts
    
    s1=np.sqrt(p1)
    s2=np.sqrt(p2)

    y,yerr=compute_Ratios(p1,p2,s1,s2)
        

    return x,y,yerr

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
       filename=common_path+mca_file[i]            
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


def read_allCMOS(common_path,cmos_eventsFiles,cmos_livetimes,  binsSdd,ax2):
   # ax2, to draw individual spectra

    XBINS=2822
    YBINS=4144  
    REBINXY=2.
     
    fig5=plt.figure(5)
    ax=fig5.subplots(1,3)

    
    fig51=plt.figure(51)
    ax51=fig51.subplots(1,1)

    xbins2d=int(XBINS/REBINXY)
    ybins2d=int(YBINS/REBINXY)
    # retta calibrazione cmos
    calP0=-0.003201340833319255
    calP1=0.003213272145961988
 
    counts=[]
    counts_all=0.
    livetime_all=0
    times_all=np.array([])
    bins=0
    min_time=0
    for i in range(0,len(  cmos_eventsFiles)):

        f=common_path+cmos_eventsFiles[i]
        print("reading: ",f)
        w, x,y,size,times=al.retrive_vectors3(f)
        print("len w =",w)
        energies=w*calP1+calP0
        #taglio spaziale!!!!
        livetime=cmos_livetimes[i]
        mask_pos=np.where((x>1310)&(x<1670)&(y<2230)&(y>1870)&(energies>0.20) )

        if(i==0):
               min_time=min(times[mask_pos])
        
        times_all=np.append(times_all,times[mask_pos])
        print("TIMES_ALL:",times_all)
        #times_all.append(times)
        
        

        #countsClu, binsE = np.histogram( energies[mask_pos]  , bins =len(binsSdd)-1, range = (binsSdd[0],binsSdd[-1]) )
        countsClu, binsE = np.histogram( energies[mask_pos], bins =len(binsSdd)-1, range = (binsSdd[0],binsSdd[-1]) )
               
        ax2.hist(binsE[:-1],bins=binsE ,weights=countsClu/livetime, histtype='step', label="cmos")
        #plt.legend()

        # plot xy:
        #plt.figure(i+10)
        counts2dClu,  xedges, yedges= np.histogram2d(x[mask_pos],y[mask_pos],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
        counts2dClu=   counts2dClu.T
        #ax=fig5.subplots(2,2)
        im=ax.flatten()[i].imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])


        avarage_rate=len( energies[mask_pos])/livetime
        avarage_rateErr=(len( energies[mask_pos])**0.5)/livetime
        print("=========>>>>> avarage_rate=",avarage_rate," +-",avarage_rateErr)
        
        ax51.errorbar(np.mean(times[mask_pos])-min_time,avarage_rate,xerr=(max(times)-min(times))/2.,yerr= avarage_rateErr,fmt='ob')
        
        if i==0:
           counts_all=countsClu
        else:
            counts_all+=countsClu 
        livetime_all+=livetime
    plt.title('cmos')

    cmos_stability(times_all,ax51)

    return binsE,counts_all,livetime_all,times_all


def cmos_stability(times_all,ax):

    print(times_all)
    
    maxtime=max(times_all)
    mintime=min(times_all)
    print("maxtime=",maxtime,' nintime=',mintime)
    n_bins=300
    binw=(maxtime-mintime)/n_bins
    
    
    countsTime, binsTime = np.histogram(times_all-mintime, bins =n_bins, range = (0,maxtime-mintime))
    #plt.figure(20)                               
    ax.hist(binsTime[:-1],bins=binsTime ,weights=countsTime/binw, histtype='step')
    ax.set_title('times cmos')

 
    

if __name__ == "__main__":

    common_path='/home/maldera/IXPE/XCF/data/MXR_15genn2026/'
    #dati sdd"
    mca_file=['misura_1_300s.mca',  'misura_2_600s.mca',  'misura_3_600s.mca']

    plt.figure(1)
    binsSdd,counts_all,livetime_all=read_allSdd(common_path+'sdd/', mca_file)
               
    #plot SDD
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.title('SDD')
    plt.hist(binsSdd[:-1],bins=binsSdd ,weights=counts_all/livetime_all, histtype='step', label="spettro somma")
    plt.legend()  
    plt.title('SDD')

    ### read CMOS data:
    list_name='events_list_pixCut10sigma_CLUcut_10sigma_v3.npz'
    cmos_eventsFiles=['/spot1/'+list_name, '/spot2/'+list_name, 'spot3/'+list_name]
    exposure=0.3  #300 ms
    cmos_livetimes=[1000*exposure,3000*exposure,3000*exposure]

    
    
    fig2=plt.figure(2)
    ax2 = fig2.subplots()
    binsCmos,counts_allCmos,livetimeCmos,times_all= read_allCMOS(  common_path, cmos_eventsFiles,cmos_livetimes,binsSdd,ax2)
    ax2.hist(binsCmos[:-1],bins=binsCmos ,weights=counts_allCmos/livetimeCmos, histtype='step', label="spettro somma")
    ax2.legend()  



    #cmos stability
    fig3=plt.figure(3)
    ax3=fig3.subplots(1,1)
    cmos_stability(times_all,ax3)



    
    plt.legend()
    plt.show()
    exit()




    
   # print("len(binsSdd=",len(binsSdd))
   # print("LEN binsCmos=",len(binsCmos))
  

    #rebinno gli istogrammi
    psdd=histogramSimo()
    psdd.counts=counts_all
    psdd.bins=binsSdd

        
    pcmos=histogramSimo()
    pcmos.counts=counts_allCmos
    pcmos.bins=binsCmos

   
    
    #pcmos.rebin(100)
    #psdd.rebin(100)
    pippo=  pcmos.counts.copy() #!!!!!!
    binCenters=pcmos.bins[0:-1]+(pcmos.bins[1]-pcmos.bins[0])/2.
    corr_counts=correct_spectrumConst2(pippo,1) # nuovo rebin!!!!!! 
    #corr_counts=pippo

   # pcmos.rebin(100)
   # psdd.rebin(100)
    
    
    print("len dopoRebin (binsSdd=",len(psdd.bins))
    print("LEN dopo rebin  binsCmos=",len(pcmos.bins))
  
    #corr_counts=correct_spectrumLin(pippo,binCenters)

    
    pcmosCorr=histogramSimo()
    pcmosCorr.counts=corr_counts
    pcmosCorr.bins=  pcmos.bins
    
    pcmos.rebin(1)
    psdd.rebin(1)
    pcmosCorr.rebin(1)

    x,y,err= compute_HistRatios(pcmos,psdd)
    xCorr,yCorr,errCorr= compute_HistRatios(pcmosCorr,psdd)
   
    plt.figure(4)

    print("livetime sdd=",livetime_all)
    print("livetime cmos=",livetimeCmos)
   
    
    ratioLivetime=livetime_all/livetimeCmos
    plt.errorbar(x,y*ratioLivetime,yerr=err*ratioLivetime,fmt='or')
    plt.errorbar(xCorr,yCorr*ratioLivetime,yerr=errCorr*ratioLivetime,fmt='ob')
    plt.xlim(1.8,7)
    plt.ylim(0,1)
    
    plt.grid()
    
    fig=plt.figure(3)
    ax = fig.subplots()
    pcmos.counts=  pcmos.counts/livetimeCmos
    pcmos.plot(ax,"cmos")

    pcmosCorr.counts=  pcmosCorr.counts/livetimeCmos
    pcmosCorr.plot(ax,"cmosCorrected")

    
    psdd.counts=  psdd.counts/livetime_all
    psdd.plot(ax,"sdd")
    plt.legend()
    plt.show()


    


