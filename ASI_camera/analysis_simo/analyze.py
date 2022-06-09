import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT
from pedestal import bg_map

shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/source/'
bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/bkg/'
create_bg_map=False


# compute pedestal files:
if create_bg_map==True:
    bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=0 )


# inizio analisi...
pedfile=bg_shots_path+'mean_ped.fits'
mean_ped=al.read_image(pedfile) # non divido per 4. il ped e' gia' stato diviso alla lettura delle immagini 

f=glob.glob(shots_path+"/*.FIT")
#creo un istogramma vuoto
x=[]
countsAll,bins=np.histogram(x,bins=int(65536/4)  ,range=(0,65536/4)  )
# creo histo root
h1=ROOT.TH1F('h1','',16384,0,16384)

#aa=[[0, 0]]
#supp_coordsAll=np.empty((0,2))
#supp_weightsAll=np.empty(0)
#x_pix=np.empty(0)
#y_pix=np.empty(0)

n=0
#print("shape iniziale=",supp_coordsAll.shape)
#print ("supp_coordsAll inizila=",supp_coordsAll)

for image_file in f:

    print('===============>>>  n=',n)
    image_data=al.read_image(image_file)/4.
    # subtract bkg:
    image_data=image_data-mean_ped
    flat_image=image_data.flatten()
    #riempio histo
    #counts_i,bins_i=np.histogram(flat_image,bins=int(65536/4)  ,range=(0,65536/4)  )
    #countsAll=countsAll+counts_i
    #root histo
    w=np.ones(len(flat_image))
    h1.FillN(len(flat_image), flat_image, w)

    if n>400:
        break
    n=n+1


h1.Draw()    
