import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import glob
import sys
sys.path.insert(0, '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/libs')
import utils as al
from pedestal import bg_map


    
shots_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Senza vetro/1s 120g sv/'
bg_shots_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Senza vetro/bg 1s 120g sv/'
create_bg_map = False


if create_bg_map == True:
    bg_map(bg_shots_path, bg_shots_path + 'mean_ped.fits', bg_shots_path + 'std_ped.fits', draw = 0 )


# inizio analisi...
pedfile  = bg_shots_path + 'mean_ped.fits'
mean_ped = al.read_image(pedfile)
pedSigmafile  = bg_shots_path + 'std_ped.fits'
rms_ped = al.read_image(pedSigmafile)

f = glob.glob(shots_path + "/*.FIT")

# creo istogramma 1d vuoto...
x = []
countsAll, bins = np.histogram(x, bins = int(65536/4), range = (0,65536/4))
countsAllClu, bins = np.histogram(x, bins = int(65536/4), range = (0,65536/4))
countsAllOnes, bins = np.histogram(x, bins = int(65536/4), range = (0,65536/4))


# creo histo2d vuoto:
countsAll2d, xedges, yedges = np.histogram2d(x, x, bins = [141, 207], range = [[0, 2822], [0, 4144]])
countsAll2dClu, xedges, yedges = np.histogram2d(x, x, bins = [141, 207], range = [[0, 2822], [0, 4144]])
countsAllcg, xedges, yedges = np.histogram2d(x, x, bins = [141, 207], range = [[0, 2822], [0, 4144]])    #istogramma con i centres of gravity

zero_img = np.zeros((2822, 4144))
image_SW = np.zeros((2822, 4144))

# MASCHERA PIXEL RUMOROSI
#mySigmaMask=np.where( (rms_ped>10)&(mean_ped>500) )
mySigmaMask = np.where( (rms_ped>3) )

n = len(f)
print('len(f) = ' + str(len(f)))

n_saved_files = 0

#np array vuoti a cui appendo..
x_all = np.empty(0)
y_all = np.empty(0)

x_allClu = np.empty(0)
y_allClu = np.empty(0)

x_cg = np.empty(0)
y_cg = np.empty(0)

n = 0.

for image_file in f:
  #  print(n," --> ", image_file)
    if n%10 == 0:
         frac = float(n/len(f)) * 100.
         print(" processed ",n," files  (  %.2f %%)" %frac )
    n = n + 1
    image_data = al.read_image(image_file)/4.
    image_data = image_data - mean_ped
    
    image_data[mySigmaMask] = 0
    
    image_SW = image_SW + image_data
    flat_image = image_data.flatten()
    counts_i, bins_i = np.histogram(flat_image, bins = int(65536/4), range = (0,65536/4))
    countsAll = countsAll + counts_i

    #####################33
    #experimental....
      
    supp_coords, supp_weights = al.select_pixels2(image_data, 25)
   # print (supp_coords.transpose())
    trasposta = supp_coords.transpose()

    # salvo posizioni che superano la selezione
    x_all = np.append(x_all, trasposta[0])
    y_all = np.append(y_all, trasposta[1])
    # istogramma 2d
 #   counts2d,  xedges, yedges= np.histogram2d(trasposta[0],trasposta[1],bins=[141,207 ],range=[[0,2822],[0,4144]])
 #   countsAll2d=countsAll2d + counts2d

    # test clustering.... # uso v2 per avere anche le posizioni
    w_clusterAll, clu_coordsAll, clu_lenghts, cg_coords = al.clustering_v2(supp_coords, supp_weights)
    w_clusterAll = np.array(w_clusterAll)
    
    #al.how_it_works(clu_coordsAll, clu_lenghts, supp_coords)    #script per vedere se il clustering sta funzionando come ci aspettiamo
    
    clu_trasposta = clu_coordsAll.transpose()
    
    x_allClu = np.append(x_allClu, clu_trasposta[0])
    y_allClu = np.append(y_allClu, clu_trasposta[1])
    
    
    cg_coords_t = cg_coords.transpose()
    
    counts2dClu, xedges, yedges = np.histogram2d(clu_trasposta[0] , clu_trasposta[1], bins=[141,207], range=[[0,2822], [0,4144]])
    countsAll2dClu = countsAll2dClu + counts2dClu
    
    countsCG, xedges, yedges = np.histogram2d(cg_coords_t[0], cg_coords_t[1], bins = [141,207], range = [[0,2822], [0,4144]])
    countsAllcg = countsAllcg + countsCG
    
    countsOnes_i, bins_i = np.histogram(w_clusterAll[clu_lenghts == 1], bins = int(65536/4), range = (0,65536/4))
    countsAllOnes = countsAllOnes + countsOnes_i
    
    countsClu_i, bins_i = np.histogram(w_clusterAll, bins = int(65536/4), range = (0,65536/4))
    countsAllClu = countsAllClu +  countsClu_i
    
    if(n == 10):
        break
    

plt.figure()
plt.imshow(image_SW, cmap = 'plasma')
plt.colorbar()




###########
# plot immagine
#fig2, ax2 = plt.subplots()
#plt.plot(x_all, y_all, 'sr', alpha = 0.3, ms = 10)
#plt.plot(x_allClu, y_allClu, 'sg', alpha = 1, markerfacecolor = 'none', ms = 11)


fig, ax = plt.subplots()
#plt.hist2d(x_all,y_all,bins=[141,207 ],range=[[0,2822],[0,4144]] )
countsAll2dClu = countsAll2dClu.T
plt.imshow(countsAll2dClu, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()
plt.title("Rebin")

#image_SW = image_SW / n
#flat_image = image_SW.flatten()

# save figures
#al.write_fitsImage(countsAll2dClu, shots_path+'imageCUL_cut25.fits'  , overwrite = "True")


# plot spettro
fig1, h1 = plt.subplots()
h1.hist(bins[:-1], bins = bins, weights = countsAll, histtype = 'step', label = "raw")
h1.hist(bins[:-1], bins = bins, weights = countsAllClu, histtype = 'step', label = 'CLUSTERING')
h1.hist(bins[:-1], bins = bins, weights = countsAllOnes, histtype = 'step', label = 'Just Ones')
plt.legend()


#plot gaussiana NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mu, std = norm.fit(countsAllOnes)
xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

fig2, gauss = plt.subplots()
plt.plot(x, p, 'k', linewidth = 2)
plt.title("Gaussian on Just Ones")

#NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

fig3, h3 = plt.subplots()
countsAllcg = countsAllcg.T
plt.imshow(countsAllcg, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()
plt.title("Centre of gravity")

# save histos
#np.savez(shots_path+'spectrum_all_raw', counts = countsAll, bins = bins)
#np.savez(shots_path+'spectrum_allCLU_cut25', counts = countsAllClu, bins = bins)



plt.show()

