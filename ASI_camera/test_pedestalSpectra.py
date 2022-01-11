from astropy.io import fits as pf
import numpy as np
from matplotlib import pyplot as plt
import glob



def read_image(nomefile):
   data_f = pf.open(nomefile, memmap=True)
   data_f.info()
   image_data = pf.getdata(nomefile, ext=0)/4.

   return image_data
   

def plot_image(image_data):
    plt.figure()
    plt.imshow(image_data, cmap='plasma')
    plt.colorbar()
    plt.show()


def isto_all(image_data):
    flat_image=image_data.flatten()
    fig, ax = plt.subplots()
    ax.hist(flat_image, bins=int(65536/4), range=(0,65536/4)   , alpha=1, histtype='step')
    mean=flat_image.mean()
    rms=flat_image.std()
    s='mean='+str(round(mean,3))+"\n"+"RMS="+str(round(rms,3))
    ax.text(0.7, 0.9, s,  transform=ax.transAxes,  bbox=dict(alpha=0.7))
    plt.show()


def plot_pixel_dist(file_list,pixel):    

   myVal=[]
   for image_file in file_list:
        image_data=read_image(image_file)
        myVal.append(image_data[pixel[0]][pixel[1]])
        print("val = ",image_data[pixel[0]][pixel[1]])
        
   npVal=np.array(myVal)
   isto_all(npVal)
        

nomefile='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-20_14_18_02Z/2021-12-20-1418_0-CapObj_0000.FIT'  # buio, 40us, 350gain, 50,50wb, 80 offset 
image_data=read_image(nomefile)
nx=image_data.shape[0]
ny=image_data.shape[1]

print ("shape=",image_data.shape," ",nx,"  ",ny)


# path alle immagini bg:
bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-20_14_22_31Z/'
# lista file immagini:
f=glob.glob(bg_shots_path+"/*.FIT")

# array somma (ogni pixel contine la somma... )
allSum=np.zeros((nx,ny),dtype=np.int16 )
# array somma^2 (ogni pixel sum(x_i^2)... )
allSum2=np.zeros((nx,ny),dtype=np.int16 )



n=0.
for image_file in f:
   n=n+1.
   print(n," --> ", image_file) 
   image_data=read_image(image_file)
   allSum=allSum+ image_data
   allSum2=allSum2+ image_data**2
  

mean=allSum/n   
plot_image(mean)

# write image w mean pedestal

hdu = pf.PrimaryHDU(mean)
hdu.writeto('new2.fits')

std=(allSum2/n-mean**2)**0.5

plot_image(std)

isto_all(image_data)

plot_pixel_dist(f,[100,100] )  





