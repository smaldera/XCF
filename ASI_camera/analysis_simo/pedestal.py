import sys

import numpy as np
from matplotlib import pyplot as plt
import glob
sys.path.insert(0, '../libs')
import utils as al

import ROOT


rootObjects=[]


def plot_pixel_dist(file_list,pixel):    

   myVal=[]
   for image_file in file_list:
        image_data=al.read_image(image_file)/4.
        myVal.append(image_data[pixel[0]][pixel[1]])
        print("val = ",image_data[pixel[0]][pixel[1]])
        
   npVal=np.array(myVal)
   al.isto_all(npVal)
        


def bg_map(bg_shots_path,outMeanPed_file, outStdPed_file, ny=4144,nx=2822, draw=1, hist_pixel=None ):

  # lista file immagini:
   f=glob.glob(bg_shots_path+"/*.FIT")

   print ("pedestals from :", bg_shots_path)
     
   if hist_pixel!=None:
       print('plotting histogram for pixel:',hist_pixel[0], " ",hist_pixel[1])
       plot_pixel_dist(f,[hist_pixel[0],hist_pixel[1]] ) # if hist_pixel differnt form null, plot the histo of that pixel and return
       return                
   
   # array somma (ogni pixel contine la somma... )
   allSum=np.zeros((nx,ny),dtype=np.int16 )
   # array somma^2 (ogni pixel sum(x_i^2)... )
   allSum2=np.zeros((nx,ny),dtype=np.int16 )

   n=0.
   for image_file in f:
      n=n+1.
      print(n," --> ", image_file)
      if n%10==0:
         frac=float(n/len(f))*100.
         print("Pedestal-> processed ",n," files  (  %.2f %%)" %frac )
      image_data=al.read_image(image_file)/4.
      allSum=allSum+ image_data
      allSum2=allSum2+ image_data**2

   # mean pedestal for each pixel    
   mean=allSum/n
   # pedestal standard deviation:
   std=(allSum2/n-mean**2)**0.5

   # write image w mean pedestal
   print ('creating pedestal files:\n')
   print ('means = ',outMeanPed_file)
   print('rms = ',outStdPed_file)
   
   al.write_fitsImage(mean,outMeanPed_file, overwrite='True' )
   al.write_fitsImage(std,outStdPed_file, overwrite='True')

   if draw:
     al.plot_image(mean)
     al.isto_all(mean)
   
     al.plot_image(std)
     al.isto_all(std)

     h_mean=al.isto_all_root(mean)
     h_mean.SetName('meanPed')
     h_mean.SetTitle('mean pedestal')
     h_mean.GetXaxis().SetTitle('ADC ch.')
    
     h_std=al.isto_all_root(std)
     h_std.SetTitle('pedestal RMS')
     h_std.GetXaxis().SetTitle('ADC ch.')
   
     c1=ROOT.TCanvas('c1','',0)
     rootObjects.append(c1)
     rootObjects.append(h_mean)
     rootObjects.append(h_std)
    
         
     c1.Divide(2)
     c1.cd(1)
     h_mean.Draw()
     c1.cd(2)
     h_std.Draw()
     c1.Update()

     # wait for stop:
     
     input('press any key to continue...')
     

#####################################

if __name__ == "__main__":
   #bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2022-05-23_10_21_08Z'
   #bg_map(bg_shots_path,'mean_ped_no_n1.fits', 'std_ped_n1.fits', draw=1 )

  # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/testFe/2022-02-11_11_56_05Z_src5sec'
  # bg_map(bg_shots_path,'mean_pedLong.fits', 'std_pedLong.fits', draw=0 )

   bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1/noGlass/G0_32us/'
   bg_map(bg_shots_path,bg_shots_path+'mean_pedLong.fits', bg_shots_path+'std_pedLong.fits', draw=1 )



  
   