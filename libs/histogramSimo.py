import numpy as np
from matplotlib import pyplot as plt
#sys.path.insert(0, '../../libs')
import fit_histogram as fitSimo
import pandas as pd
from read_sdd import  *

class histogramSimo():
      """
      primo tentativo di una classe per racchiudere tutte le funzioni che riguardano istogrammi 1D
      per ora utile solo per plottare e normalizzare
      possibili aggiunte (todo):
       - creazione con numpy passando l'array
       - salvataggio dei 2 vettori
       - calcolo media e rms
       - 
      """

      def __init__(self, counts=None, bins=None):

             self.counts=counts
             self.bins=bins
             self.sdd_liveTime=None
             self.sdd_deadTime=None
             self.sdd_fastCounts=None
             self.sdd_start=None

     # def get_from_file(self,filename):
     #       data=np.load(filename)
     #       counts=data['counts']
     #       bins=data['bins']
     #       self.counts=counts
     #       self.bins=bins

      def normalize(self, minX,maxX):      
            bin_centers=fitSimo.get_centers(self.bins)
            print("minx=",minX," max =",maxX)
            mask=np.where((bin_centers>minX)&(bin_centers<maxX))
            c2=self.counts[mask]
            print("len(self.counts)=",len(self.counts)," len bin centers=",len(bin_centers), " len(c2)=",len(c2))
            
            ka_h=np.max(c2)
            print("Ka max=",ka_h) 

            self.counts=self.counts/ka_h
            
      def plot(self,ax,labelname):

            histo=ax.hist(self.bins[:-1],bins=self.bins,weights=self.counts, histtype='step', label=labelname)



      def read_from_file(self, filename, fileFormat ):

            data_array=None
     
            if fileFormat=='sdd':
                data_array, deadTime, livetime, fast_counts, start =pharse_mca(filename)
                size=len(data_array)      
                bin_edges=np.linspace(0,size+1,size+2)[:-1] # trucco per avere i bin edges all'intero esatto
                self.counts=data_array
                self.bins=bin_edges
                self.sdd_liveTime=livetime
                self.sdd_deadTime=deadTime
                self.sdd_fastCounts=fast_counts
                self.sdd_start=start
          
            if fileFormat=='Eric_mcPherson':
                 df=pd.read_csv(filename,header=None)
                 #x=df[0].values
                 y=df[1].values
                 size=len(y)      
                 bin_edges=np.linspace(0,size+1,size+2)[:-1] 
                 self.counts=y
                 self.bins=bin_edges


            if  fileFormat=='npz':
                 data=np.load(filename)
                 counts=data['counts']
                 bins=data['bins']
                 self.counts=counts
                 self.bins=bins

            if fileFormat=='sddnpz':
                 data_array, deadTime, livetime, fast_counts,slowcounts,  start,stop= pharse_ssdnpz(filename)
                 size=len(data_array)      
                 bin_edges=np.linspace(0,size+1,size+2)[:-1] 
                 print ("bin edges=",bin_edges)
                 self.counts=data_array
                 self.bins=bin_edges
                 self.sdd_liveTime=livetime
                 self.sdd_deadTime=deadTime
                 self.sdd_fastCounts=fast_counts
                 self.sdd_start=start  
                 
      def rebin(self,n):
          
         rebinned_counts=[]
         rebinned_bins=[]
         rebinned_bins.append(self.bins[0])

         summed_counts=0.
         n_sum=0
         print ("len_counts=",len(self.counts))
         for i in range (0,len(self.counts)):
               
               if n_sum<n:
                     summed_counts=summed_counts+self.counts[i]
                  #   print("counts[i]=",self.counts[i])
                     n_sum=n_sum+1
                   #  print("i=", i," counts=", self.counts[i]," bin max=",self.bins[i+1]," sum= ",summed_counts," n_sum=",n_sum)
                     if n_sum==n:
                           rebinned_counts.append(summed_counts)
                           rebinned_bins.append(self.bins[i+1])
                    #       print("n_sum=n!!!!!!!!!!!!!!!!!", " reninned=",summed_counts," bin right=",self.bins[i+1])
                           summed_counts=0.
                           n_sum=0
                           if (len(self.counts)-i)<n:
                                # print ("i=",i," len-i=",len(self.counts)-i," stop here"  )
                                 break

         self.counts=np.array(rebinned_counts)
         self.bins=np.array(rebinned_bins)
         

                           
      def save_npz(self,filename):
               np.savez(filename, counts = self.counts,  bins = self.nins)
               


               
