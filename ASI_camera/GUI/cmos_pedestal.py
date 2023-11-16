import numpy as np
import glob
#sys.path.insert(0, '../../libs')
import utils_v2 as al
from tqdm import tqdm


def plot_pixel_dist(file_list,pixel):    

   myVal=[]
   for image_file in file_list:
        image_data=al.read_image(image_file)/4.
        myVal.append(image_data[pixel[0]][pixel[1]])
        #print("val = ",image_data[pixel[0]][pixel[1]])
        
   npVal=np.array(myVal)
   al.isto_all(npVal)
        


def bg_map(bg_shots_path, outMeanPed_file, outStdPed_file, OutPath, OutName, ny=4144,nx=2822, hist_pixel=None):

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
   for image_file in tqdm(f):
      n=n+1.
     # print(n," --> ", image_file)
      #if n%10==0:
      #   frac=float(n/len(f))*100.
      #   print("Pedestal-> processed ",n," files  (  %.2f %%)" %frac )
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
   
   al.plot_image(mean, OutPath, OutName+'_mean_plot.png')
   al.isto_all(mean, OutPath, OutName+'_mean_hist.png')

   al.plot_image(std, OutPath, OutName+'_std_plot.png')
   al.isto_all(std, OutPath, OutName+'_std_hist.png')
   
   input('press any key to continue...')
