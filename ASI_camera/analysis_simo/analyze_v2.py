import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils_v2 as al
from pedestal import bg_map


    
shots_path = '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120/'
bg_shots_path = '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120_bg/'
create_bg_map = False
NBINS=16384  # n.canali ADC (2^14)




if create_bg_map == True:
    bg_map(bg_shots_path, bg_shots_path + 'mean_ped.fits', bg_shots_path + 'std_ped.fits', draw = 0 )


# inizio analisi...
# leggo files pedestal (mean e rms)
pedfile  = bg_shots_path + 'mean_ped.fits'
mean_ped = al.read_image(pedfile)
pedSigmafile  = bg_shots_path + 'std_ped.fits'
rms_ped = al.read_image(pedSigmafile)


f = glob.glob(shots_path + "/*.FIT")

# creo istogrammi 1d vuoti
x = []
countsAll, bins = np.histogram(x, bins = 2*NBINS, range = (-NBINS,NBINS))
countsAllClu, bins = np.histogram(x,  bins = 2*NBINS, range = (-NBINS,NBINS))


# creo histo2d vuoto:
countsAll2dClu,  xedges, yedges= np.histogram2d(x,x,bins=[141,207 ],range=[[0,2822],[0,4144]])

zero_img = np.zeros((2822, 4144))
image_SW = np.zeros((2822, 4144))


# MASCHERA PIXEL RUMOROSI 
#mySigmaMask=np.where( (rms_ped>10)&(mean_ped>500) )
mySigmaMask=np.where( (rms_ped>10) )


#np array vuoti a cui appendo le coordinate
x_all=np.empty(0)
y_all=np.empty(0)
x_allClu=np.empty(0)
y_allClu=np.empty(0)

n=0.
# inizio loop sui files
for image_file in f:
    #print(n," --> ", image_file)
    if n%10==0:
         frac=float(n/len(f))*100.
         print(" processed ",n," files  (  %.2f %%)" %frac )
    n=n+1

    # read image:
    image_data = al.read_image(image_file)/4.
    # subtract pedestal:
    image_data = image_data -  mean_ped #
    #applica maschera
    image_data[mySigmaMask]=0 # maschero tutti i pixel con RMS pedestal > soglia 
   
    image_SW = image_SW + image_data
    flat_image = image_data.flatten()

    # spettro "raw"
    counts_i, bins_i = np.histogram(flat_image,  bins = 2*NBINS, range = (-NBINS,NBINS) ) 
    countsAll = countsAll + counts_i
    
    #################
    #CLUSTERING
    # applico selezione su carica dei pixel
    #supp_coords, supp_weights=al.select_pixels2(image_data, 25)
    supp_coords, supp_weights=al.select_pixels_RMS(image_data,rms_ped,5)
   
    # test clustering.... # uso v2 per avere anche le posizioni
    w_clusterAll, clu_coordsAll=al.clustering_v2(supp_coords,supp_weights) # TODO: supp_coords sono tutte le coordinate di tutti i pixel dei cluster. ci vorrebbe una media pesata per ogni cluster  
    clu_trasposta= clu_coordsAll.transpose()
    
    # x_allClu=np.append(x_allClu,clu_trasposta[0])
    # y_allClu=np.append(y_allClu,clu_trasposta[1])

    # istogramma 2d dopo clusterimg
    counts2dClu,  xedges, yedges= np.histogram2d(clu_trasposta[0],clu_trasposta[1],bins=[141,207 ],range=[[0,2822],[0,4144]])
    countsAll2dClu=countsAll2dClu+ counts2dClu

    # istogramma spettro dopo il clustering
    countsClu_i, bins_i = np.histogram(  w_clusterAll, bins = 2*NBINS, range = (-NBINS,NBINS) )
    countsAllClu = countsAllClu +  countsClu_i
    
    if n>10:
        break


###########
# plot immagine
fig2, ax2 = plt.subplots()
#plt.plot(x_all,y_all,'or',alpha=1)
#plt.plot(x_allClu,y_allClu,'sg',alpha=0.3,markerfacecolor='none',ms=11)
countsAll2dClu=   countsAll2dClu.T
plt.imshow(countsAll2dClu, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()
plt.title('hit pixels (rebinned)')

# plot immagine SOMMA
fig3, ax3 = plt.subplots()
#image_SW = image_SW / n
plt.imshow(image_SW)
plt.colorbar()
plt.title('SUM of all frames')



# plot spettro
fig, h1 = plt.subplots()
h1.hist(bins[:-1], bins = bins, weights = countsAll, histtype = 'step',label="raw")
h1.hist(bins[:-1], bins = bins, weights = countsAllClu, histtype = 'step',label='CLUSTERING')
plt.legend()
plt.title('spectra')


# spettro immagine somma:
fig4, h4 = plt.subplots()
h4.hist(image_SW.flatten(),  bins = 2*int(65536), range = (-65536,65536) , histtype = 'step',label="spettro immagine somma")
plt.legend()
plt.title('spettro immagine somma')

# save histos
np.savez(shots_path+'spectrum_all_raw_rmsCut10', counts = countsAll, bins = bins)
np.savez(shots_path+'spectrum_allCLU_cut5sigma_rmsCut10', counts = countsAllClu, bins = bins)
np.savez(shots_path+'spectrum_SUM', counts =image_SW.flatten(), bins = bins)


# save figures
al.write_fitsImage(countsAll2dClu, shots_path+'imageCUL_cut5sigma_1pixel.fits'  , overwrite = "True")
al.write_fitsImage(image_SW, shots_path+'imageSUM.fits'  , overwrite = "True")



plt.show()

