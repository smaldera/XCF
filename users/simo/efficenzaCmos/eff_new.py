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

def weighted_mean(xi, xi_err):

       somma=(np.sum(xi/(xi_err**2)))/(np.sum(1./xi_err**2))
       somma_err=(1./(np.sum(1./xi_err**2 )))**0.5

       return somma, somma_err
       


def read_allSdd(common_path, mca_file,Ecut=0.2):
    # legge files, plotta, restituosce istogremmi e somma (normalizzati per livetime) 
       
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    
    counts_all=0.
    livetime_all=0
    rates=[]
    rates_err=[]

    figSdd=plt.figure("SDD spectra")
    axSdd=figSdd.subplots(1,1)

    histo_list=[]
    
    for i in range (0,len(mca_file)):
   

       p=histogramSimo()
       filename=common_path+mca_file[i]            
       p.read_from_file(filename, 'sdd' )
       # calibrazione energia
       p.bins=p.bins*calP1+calP0

       histo_list.append(p)
              
       axSdd.hist(p.bins[:-1],bins=p.bins ,weights=p.counts/p.sdd_liveTime, histtype='step', label=mca_file[i][0:-4])
       
       
       livetime_all+=p.sdd_liveTime
       if i==0:
           counts_all=p.counts
       else:
           counts_all=p.counts+counts_all # OKKIO!!! se uso +=  fa casini nell'array p.counts originale!!
               
    #plot sum histo

    pSum=histogramSimo( counts=counts_all, bins= histo_list[0].bins)
    pSum.sdd_liveTime=livetime_all

    axSdd.hist(p.bins[:-1],bins=p.bins ,weights=pSum.counts/pSum.sdd_liveTime, histtype='step', label="sum")
    
    axSdd.set_xlabel('keV')
    axSdd.set_ylabel('counts/s [Hz]')
    axSdd.set_yscale("log") 
    axSdd.legend()        
           
    
    return  histo_list, pSum


def read_allCMOS(common_path,cmos_eventsFiles,cmos_livetimes,  binsSdd):
    

    XBINS=2822
    YBINS=4144  
    REBINXY=2.
     
    figCMOS=plt.figure('cmos')
    ax=figCMOS.subplots(1,3)

    
    figCmosE=plt.figure('cmosEnergy')
    axCmosE=figCmosE.subplots(1,1)

    figCmosStability=plt.figure('CmosStability')
    axCmosSt=figCmosStability.subplots(1,1)

    
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
    rates=[]
    ratesErr=[]


    histo_list=[]
    
    for i in range(0,len(  cmos_eventsFiles)):

        f=common_path+cmos_eventsFiles[i]
        print("reading: ",f)
        w, x,y,size,times=al.retrive_vectors3(f)
       
        energies=w*calP1+calP0
        #taglio spaziale!!!!
        livetime=cmos_livetimes[i]
        mask_pos=np.where((x>1310)&(x<1670)&(y<2230)&(y>1870)&(energies>0.1) )

        if(i==0):
               min_time=min(times[mask_pos])
        
        times_all=np.append(times_all,times[mask_pos])
       
       
        
        countsClu, binsE = np.histogram( energies[mask_pos], bins =len(binsSdd)-1, range = (binsSdd[0],binsSdd[-1]) )
        axCmosE.hist(binsE[:-1],bins=binsE ,weights=countsClu/livetime, histtype='step', label="cmos")
       

        #creo istogrammi... e appendo
        pCmos=histogramSimo( counts=countsClu, bins= binsE)
        pCmos.sdd_liveTime=livetime
        histo_list.append(pCmos)
        
        # plot xy:
        
        counts2dClu,  xedges, yedges= np.histogram2d(x[mask_pos],y[mask_pos],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
        counts2dClu=   counts2dClu.T
        
        im=ax.flatten()[i].imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])


        avarage_rate=len( energies[mask_pos])/livetime
        avarage_rateErr=(len( energies[mask_pos])**0.5)/livetime
        #print("=========>>>>> avarage_rate=",avarage_rate," +-",avarage_rateErr)
        rates.append(avarage_rate)
        ratesErr.append(avarage_rateErr)
        axCmosSt.errorbar(np.mean(times[mask_pos])-min_time,avarage_rate,xerr=(max(times)-min(times))/2.,yerr= avarage_rateErr,fmt='ob')
        
        if i==0:
              counts_all= countsClu              
        else:
            counts_all=counts_all+countsClu
                                     
        livetime_all+=livetime
   

        
    #creo histo sum
    pCmosSum=histogramSimo( counts=counts_all, bins= binsE)
    pCmosSum.sdd_liveTime=livetime_all
    axCmosE.hist(binsE[:-1],bins=binsE ,weights=counts_all/livetime_all, histtype='step', label="sum")
    
    # labels assi etc.
    axCmosE.set_xlabel('energy [keV]')
    axCmosE.set_ylabel('counts/s [Hz]')
    axCmosE.set_yscale("log") 
    axCmosE.legend()

    axCmosSt.set_xlabel('elapsed time[s]')
    axCmosSt.set_ylabel('rate CMOS')
    axCmosSt.grid()
   

    return  histo_list,  pCmosSum


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



def getRates_fromHistoSimo(p,Ecut=0.2):

       binCenters=p.bins[0:-1]+(p.bins[1]-p.bins[0])/2.
       mymask=np.where(binCenters>Ecut)
       #print("somma totale=",np.sum(p.counts)," E>0.2=>",np.sum(p.counts[mymask])," livetime=",p.sdd_liveTime )
       rate=np.sum(p.counts[mymask])/p.sdd_liveTime
       rate_err=(np.sum(p.counts[mymask])**0.5)/p.sdd_liveTime

       print("rate=",rate," err=",rate_err)
       return rate, rate_err    

def get_sddRates(histoSdd_list,pSum, Ecut=0.2):

       rates=[]
       rates_err=[]

       
       
       for histo in histoSdd_list:
           rate, rate_err= getRates_fromHistoSimo(histo,Ecut)
           rates.append(rate)
           rates_err.append(rate_err)

          
        
       rateSum, rateSum_err= getRates_fromHistoSimo(pSum,Ecut)    

       return  np.array(rates),np.array( rates_err),rateSum, rateSum_err

if __name__ == "__main__":

    common_path='/home/maldera/IXPE/XCF/data/MXR_15genn2026/'
    #dati sdd"
    mca_file=['misura_1_300s.mca',  'misura_2_600s.mca',  'misura_3_600s.mca']

    histoSdd_list,hSum_sdd=read_allSdd(common_path+'sdd/', mca_file)
    rates_sdd, rates_sddErr,rate_sddSum, rate_sddSumErr= get_sddRates(histoSdd_list,hSum_sdd)

    for i in range(0,len(rates_sdd)):
       print("---->>> rates SDD",  rates_sdd[i]," +-  ", rates_sddErr[i])  
    print("===>>>> rates sdd SUM",  rate_sddSum," +-  ", rate_sddSumErr)  
    
        
    print('\n ===================================================== \n')
    
    ### read CMOS data:
    list_name='events_list_pixCut10sigma_CLUcut_10sigma_v3.npz'
    cmos_eventsFiles=['/spot1/'+list_name, '/spot2/'+list_name, 'spot3/'+list_name]
    exposure=0.3  #300 ms
    cmos_livetimes=[1000*exposure,3000*exposure,3000*exposure]

    
    binsSdd= histoSdd_list[0].bins
    histoCmos_list, hSum_cmos = read_allCMOS(  common_path, cmos_eventsFiles,cmos_livetimes,binsSdd)


    rates_cmos, rates_cmosErr,rate_cmosSum, rate_cmosSumErr= get_sddRates(histoCmos_list,hSum_cmos)
    for i in range(0,len(rates_cmos)):
       print("---->>> rates cmos",  rates_cmos[i]," +-  ", rates_cmosErr[i])  
    print("===>>>> rates rates cmos SUM",  rate_cmosSum," +-  ", rate_cmosSumErr)  

  
     
    r, rErr=compute_Ratios(rates_cmos,rates_sdd,rates_cmosErr,rates_sddErr)
    print("RATIOS=",r)
    print("err=",rErr)                        
                             
    somma, somma_err=weighted_mean(r, rErr)
    print("EFF pesata=",somma," somma_err",somma_err)


    

    #calcolare eff golbale
    r_global,rErr_global=compute_Ratios(rate_cmosSum,rate_sddSum,rate_cmosSumErr,rate_sddSumErr)
    
    print("EFF globale=",r_global," +- "  , rErr_global)

                             
    plt.show()





    


