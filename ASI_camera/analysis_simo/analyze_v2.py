import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
from pedestal import bg_map


def retrive_histo(nomefile):
    data = np.load(nomefile)
    counts = data['counts']
    bins = data['bins']
    fig, ax = plt.subplots()
    ax.hist(bins[:-1], bins = bins,weights=counts, histtype = 'step')
    plt.show()


    
shots_path = '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120_bg/'
bg_shots_path = '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120_bg/'

create_bg_map = False


if create_bg_map == True:
    bg_map(bg_shots_path, bg_shots_path + 'mean_ped.fits', bg_shots_path + 'std_ped.fits', draw = 0 )


# inizio analisi...
pedfile  = bg_shots_path + 'mean_ped.fits'
mean_ped = al.read_image(pedfile)

f = glob.glob(shots_path + "/*.FIT")
x = []
countsAll, bins = np.histogram(x, bins = int(65536/4), range = (0,65536/4))

fig, h1 = plt.subplots()

n = len(f)

print('len(f) = ' + str(len(f)))

zero_img = np.zeros((2822, 4144))
image_SW = np.zeros((2822, 4144))
n_saved_files = 0



n=0.
for image_file in f:
  #  print(n," --> ", image_file)
    if n%10==0:
         frac=float(n/len(f))*100.
         print(" processed ",n," files  (  %.2f %%)" %frac )
    n=n+1     
    image_data = al.read_image(image_file)/4.
    image_data = image_data - mean_ped
    image_SW = image_SW + image_data
    flat_image = image_data.flatten()
    counts_i, bins_i = np.histogram(flat_image, bins = int(65536/4), range = (0,65536/4))
    countsAll = countsAll + counts_i
    
image_SW = image_SW / n
flat_image = image_SW.flatten()



al.write_fitsImage(image_SW, shots_path+'mean_image.fits'  , overwrite = "True")
np.savez(shots_path+'spectrum_all', counts = countsAll, bins = bins)

h1.hist(bins[:-1], bins = bins, weights = countsAll, histtype = 'step')   
al.plot_image(image_SW)

plt.show()

