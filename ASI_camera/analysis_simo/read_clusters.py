

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
import pylandau

calP0=-0.003201340833319255
calP1=0.003213272145961988
XBINS=2822
YBINS=4144
NBINS=16384  # n.canali ADC (2^14)
   
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
   countsE, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))

   countsETrack_all, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))
   countsETrack, binsE = np.histogram(x, bins =n_energyBins, range = (0,100))

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
      
      if mytrack.getSize()==trackLen and abs(mytrack.corr)>min_corr:
           countsETrack, binsE = np.histogram(mytrack.getTotE(), bins =n_energyBins, range = (0,100))
           countsETrack_all=countsETrack_all+countsETrack
        
          
           if np.sum(mytrack.w[0:2])/2.>np.sum(mytrack.w[-2:])/2. and mytrack.w[0]>0:
               countsE, binsE = np.histogram(mytrack.w[n_pix], bins =n_energyBins, range = (0,100))
               countsE_all=countsE_all+countsE
               
           if np.sum(mytrack.w[0:2])/2.<=np.sum(mytrack.w[-2:])/2. and mytrack.w[-1]>0:
               countsE, binsE = np.histogram(mytrack.w[-(n_pix+1)], bins =n_energyBins, range = (0,100))                 
               countsE_all=countsE_all+countsE

   return  countsE_all,  countsETrack_all, binsE   





########################################3

###########################################


files_list=glob.glob('/home/maldera/Desktop/eXTP/data/CMOS_verticale/clusters/tracks*/*img*.npz')
tracks_list=read_allClusters(files_list)


countsE_all=0
countsETrack_all=0
binsE=0
r_all=0
totE=0
size_all=0

xpix=[]
MPV=[]

#apply Ecut and compute r_corr
r_all ,totE, size_all  = cut_allTracks(tracks_list,Ecut=0.)

SEL_SIZE=10

# loop su n_pix (ie n-esima posizione lungo la traccia, una volta fissata la lunghezza)
for n_pix in range(0,SEL_SIZE):
   
   countsE_all,  countsETrack_all, binsE =  analyze_tracks(tracks_list, n_energyBins=600,  trackLen=SEL_SIZE, min_corr=0.85, n_pix=n_pix)
   print("nPix=",n_pix)
   #plot E histogram                                  
   plt.figure(n_pix)
   plt.hist(binsE[:-1], bins =binsE, weights =countsE_all  , histtype = 'step',label='Etracks_'+str(n_pix))
   plt.title('Epixel n.  '+str(n_pix))
  

   coeff,pcov=  fit_landauHistohram(countsE_all,binsE)   
      
   x=np.arange(-0.5,2.5,0.05)
   #plt.plot(x, pylandau.langau(x, *coeff), "-")
   plt.plot(x, pylandau.landau(x, *coeff), "-")



   #x=np.arange(-1,3.5,0.005)
   #plt.plot(x, fh.landau_gausPedestal_model(x, *coeff), "-")
   #plt.plot(x, fh.landau_gausPedestal_model(x,40,0.11,0.05,1,0.4,35 ), "-")
   
   #plt.plot(x,  pylandau.landau(x, *coeff), "-")
   plt.xlim(-1,5)
   
   xpix.append(n_pix)
   MPV.append(coeff[0])
 
  
   
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

# plot tot track energy
plt.figure(60)
plt.title('track totalE')
plt.ylabel("abs(cluster lienar correlation coef.)")
plt.hist(binsE[:-1], bins =binsE, weights=countsETrack_all, histtype = 'step')
# fit total energy
coeff,pcov=  fit_landauHistohram(countsETrack_all,binsE,xmin=1,xmax=20)    
x=np.arange(1,20,0.05)
plt.plot(x, pylandau.landau(x, *coeff), "-")


mpv_track=coeff[0]
mpv_90=0.6
alpha=np.arcsin(mpv_90/mpv_track)

print("ALPHA=",alpha, " (",np.degrees(alpha),"deg)" )

# plot MPV vs n_pix
plt.figure(70)

plt.ylabel("MPV")
plt.xlabel("n pix")


depth=np.array(xpix)*4.6*np.tan(alpha)
plt.plot(depth,MPV,'ro')
#plt.legend()
plt.title('track totalE')




plt.show()
                                  
                                  

                                  
                                  
                 
