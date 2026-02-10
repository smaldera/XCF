

# aggiunger loop su diverse track size
# aggiungere parametri fit

import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import utils_v2 as al
from tqdm import tqdm

from scipy.stats import pearsonr

import fit_histogram as fh
#import pylandau

calP0=-0.003201340833319255
calP1=0.003213272145961988
XBINS=2822
YBINS=4144
NBINS=16384  # n.canali ADC (2^14)
minfit=0.3
maxfit=2

class track():
   def __init__(self, x,y,w):
      self.x = x
      self.y = y
      self.w=w
      self.corr=-100
            
   def cut_minE(self, minE):
       w_cut=np.where(self.w>minE)
       self.x=self.x[w_cut] 
       self.y=self.y[w_cut] 
       self.w=self.w[w_cut] 
   def getSize(self):
      w=self.w
     # print("len w=",len(w))
      return len(w)
   def getTotE(self):
      return np.sum(self.w)
   def ComputeCorr(self):
       self.corr, _ = pearsonr(self.x, self.y)
   
   def plot(self):
        print("x=",self.x)
        print("y=",self.y)
        print("w=",self.w)
        # creo histo2d:
        countsCharge,  xedges, yedges=       np.histogram2d(self.x,self.y,weights=self.w,bins=[XBINS, YBINS],range=[[0,XBINS],[0,YBINS]])
        countsCharge=  countsCharge.T
        plt.imshow(countsCharge, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar()
        plt.xlim(min(self.x)-10,max(self.x)+10 )
        plt.ylim(min(self.y)-10,max(self.y)+10) 
        plt.show()
        


##############3
def fit_landauHistohram(counts,binsE,xmin=-0.5,xmax=2.5):

   max_val=max(counts)
   max_index=np.argmax(counts)
   max_x=binsE[ max_index]+(binsE[1]-binsE[0])/2.
   
   print( "max_val=", max_val)
   print( "max_x=", max_x, " max_index =", max_index)
   
 
   initial_pars=[max_x,0.4,max_val]
   lims_low=[max_x-0.1,0.2,max_val-20]
   if max_x-0.1<0.45:
      lims_low[0]=0.45
   if initial_pars[0]<0.45:
       initial_pars[0]=0.45
      
   lims_up=[max_x+0.1,0.6,max_val+20]  
   coeff,pcov= fh.fit_Landau_histogram(counts,binsE,xmin=xmin,xmax=xmax,  initial_pars= initial_pars, parsBoundsLow=lims_low, parsBoundsUp=lims_up  )
   print ("coeff=",coeff)
   print ("pcov=",pcov)
  
   return  coeff,pcov

   

        
        

def read_allClusters(files_list):
   
   tracks_list=[]
   print ("loading tracks... ")
   #for myfile in files_list:
   for myfile in tqdm(files_list, colour='green'):              
       #print ("reading file: ",myfile)
       loaded=np.load(myfile)
       x=loaded['x']
       y=loaded['y']
       w=loaded['w']
       w=w*calP1+calP0
       myTrack= track(x,y,w)
       tracks_list.append(myTrack)
   return tracks_list    



def cut_allTracks(tracks_list,Ecut=0):
   print("loop on all tracks and cut E<Ecut")
   r_all=[]
   size_all=[]
   totE=[] 
   for mytrack in tracks_list:
         # applico Ecut:
         mytrack.cut_minE(Ecut)
         #calcolo_correlazione:
         mytrack.ComputeCorr()
         r_all.append(abs(mytrack.corr))
         size_all.append(mytrack.getSize() )
         totE.append(mytrack.getTotE())
   print("... done")
   return r_all,size_all,totE
 

def analyze_tracks(tracks_list,n_energyBins=1000, trackLen=7, min_corr=0.8, n_pix=0):

   print("n_energyBins=",n_energyBins)
   x = []
   countsE_all, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))
   countsE, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))   # istogramma per ogni pixel

   countsETrack_all, binsEtrack = np.histogram(x, bins =100, range = (0,100))
   countsETrack, binsEtrack= np.histogram(x, bins =100, range = (0,100))   # istogramma E totale traccia (integrando i pixel)

   r_all=[]
   size_all=[]
   totE=[]
   print("n tracks=",len(tracks_list))

   for mytrack in tracks_list:
      # applico Ecut:
      #mytrack.cut_minE(Ecut)
      #calcolo_correlazione:
      #mytrack.ComputeCorr()
      #r_all.append(abs(mytrack.corr))
      #size_all.append(mytrack.getSize() )
      #totE.append(mytrack.getTotE())

      #print("mytrack.w=",mytrack.w[0])
      #print("mytrack.get=",mytrack.w[0])
      
      if mytrack.getSize()==trackLen and abs(mytrack.corr)>min_corr:  # seleziono size e cut su correlezione
           countsETrack, binsEtrack = np.histogram(mytrack.getTotE(), bins =100, range = (0,100))
           countsETrack_all=countsETrack_all+countsETrack                                            # istogramma E totale
        
          #aggiungo l'energia all'n-esimo pixel della traccia
           if np.sum(mytrack.w[0:2])/2.>np.sum(mytrack.w[-2:])/2. and mytrack.w[0]>0:
               countsE, binsE = np.histogram(mytrack.w[n_pix], bins =n_energyBins, range = (0,100))
               countsE_all=countsE_all+countsE
               
           if np.sum(mytrack.w[0:2])/2.<=np.sum(mytrack.w[-2:])/2. and mytrack.w[-1]>0:
               countsE, binsE = np.histogram(mytrack.w[-(n_pix+1)], bins =n_energyBins, range = (0,100))                 
               countsE_all=countsE_all+countsE

   return  countsE_all,  countsETrack_all, binsE, binsEtrack   



def analize_allPix(track_list,SEL_SIZE,MIN_CORR=0.85):
   
   xpix=[]
   MPV=[]
   MPVerr=[]
   countsETrack=None
   binsE=None
   # loop su n_pix (ie n-esima posizione lungo la traccia, una volta fissata la lunghezza)
   for n_pix in range(0,SEL_SIZE):
   
      countsE_all,  countsETrack_all, binsE, binsEtrack=  analyze_tracks(tracks_list, n_energyBins=2400,  trackLen=SEL_SIZE, min_corr=MIN_CORR, n_pix=n_pix)
      print("nPix=",n_pix)
      
      #coeff0,pcov0=  fit_landauHistohram(countsE_all,binsE)
      coeff,pcov,chi2,chi2red=  fh.fit_landauHHisto_cmosMip(countsE_all,binsE,xmin=minfit,xmax=maxfit)

      #plot E histogram                                  
      plt.figure(n_pix,(13,10))
      figPixels, (axPix) = plt.subplots(1, figsize=(12,8))
      axPix.hist(binsE[:-1], bins =binsE, weights =countsE_all  , histtype = 'step',label='Etracks_'+str(n_pix))
      axPix.set_title('Epixel n.  '+str(n_pix))
      x=np.arange(minfit,maxfit,0.05)
      axPix.plot(x, fh.myLandau(x, *coeff), "-r",label='landau')
      axPix.set_xlim(-0.2,2)
      #plt.show()
      xpix.append(n_pix+1)
      MPV.append(coeff[0])
      MPVerr.append(pcov[0][0]**0.5)

   #end loop pixels
   # plot tot track energy
   print("!!!!!!!!!! FIT total track Energy =====================  ")
   figEtrack, (axEt) = plt.subplots(1, figsize=(12,8))
   axEt.set_title('track totalE+ SIZE='+str(SEL_SIZE))
   axEt.set_xlabel("track total E [keV]")
   axEt.hist(binsEtrack[:-1], bins =binsEtrack, weights=countsETrack_all, histtype = 'step')

  

   initial_pars=[ 9,    1.8,  600]
   lims_low=[5,1,500]
   lims_up=[15,4,1000]
  
   
   coeff,pcov,chi2,chi2red= fh.fit_Landau_histogram2(countsETrack_all,binsEtrack,xmin=1,xmax=20,  initial_pars= initial_pars, parsBoundsLow=lims_low, parsBoundsUp=lims_up  )

   
   print("binE=",binsEtrack)
   print("counts=",countsETrack_all)
   al.save_histo('trackE', countsETrack_all,binsEtrack)
   x=np.arange(1,20,0.05)
   axEt.plot(x, fh.myLandau(x, coeff[0],coeff[1],coeff[2]))

   
   mpv_track=coeff[0] 

   
   
   return xpix,MPV,MPVerr, mpv_track
   



########################################3

###########################################


files_list=glob.glob('/home/maldera/IXPE/XCF/data/CMOS_verticale/clusters/tracks*/img_*.npz')
tracks_list=read_allClusters(files_list)


countsE_all=0
countsETrack_all=0
binsE=0
r_all=0
totE=0
size_all=0

#apply Ecut and compute r_corr
r_all ,totE, size_all  = cut_allTracks(tracks_list,Ecut=0.)

fig1, (axFinal) = plt.subplots(1, figsize=(12,8))

#ax1.ylabel("MPV")
#ax1.xlabel("n pix")

mpv_90=0.351
MIN_CORR=0.85

for SEL_SIZE in [11]:
   print("SEL_SIZE=",SEL_SIZE)
   xpix,MPV,MPVerr,mpv_track = analize_allPix(tracks_list,SEL_SIZE, MIN_CORR=MIN_CORR)
   alpha=np.arcsin(mpv_90/mpv_track)
   print("ALPHA=",alpha, " (",np.degrees(alpha),"deg)" )
   depth=np.array(xpix)*4.6*np.tan(alpha)

   axFinal.errorbar(depth,MPV,yerr=MPVerr,fmt='o',label="size="+str(SEL_SIZE))

axFinal.legend()


   
   
 ############
plt.figure(30)
counts_r,bins_r= np.histogram(r_all, bins =100, range = (-1,1))
plt.hist(bins_r[:-1], bins =bins_r, weights=counts_r  , histtype = 'step')
#plt.legend()
plt.title('track r')

plt.figure(40)
counts_s,bins_s= np.histogram(size_all, bins =100, range = (0,100))
plt.hist(bins_s[:-1], bins =bins_s, weights=counts_s  , histtype = 'step')
#plt.legend()
plt.title('track size')


# plot r vs energy
plt.figure(50) 
plt.plot(totE,r_all,"p", alpha=0.5)
plt.xlabel("cluster E [keV]")
plt.ylabel("abs(cluster lienar correlation coef.)")






plt.show()
                                  
                                  

                                  
                                  
                 
