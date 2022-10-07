import numpy as np
from astropy.io import fits as pf
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/libs')
import utils as al
from pedestal import bg_map




shots_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Senza vetro/1/'
bg_shots_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Senza vetro/bg 1s 120g sv/'

create_bg_map = False

def retrive_histo(nomefile):
    data = np.load(nomefile)
    counts = data['counts']
    bins = data['bins']
    fig, ax = plt.subplots()
    ax.hist(bins[:-1], bins = bins,weights=counts, histtype = 'step')
    plt.show()

if create_bg_map == True:
    bg_map(bg_shots_path, bg_shots_path + 'mean_ped.fits', bg_shots_path + 'std_ped.fits', draw = 0 )


# inizio analisi...
pedfile  = bg_shots_path + 'mean_ped.fits'

mean_ped = al.read_image(pedfile)

f = glob.glob(shots_path + "*.FIT")

x = []
countsAll, bins = np.histogram(x, bins = int(65536/4), range = (0,65536/4))

n = len(f)

print('len(f) = ' + str(len(f)))

image_SW = np.zeros((2822, 4144))

for image_file in f:

    pf.open(image_file, memmap = True)
    image_data = pf.getdata(image_file, ext = 0)
    
    image_data = image_data - mean_ped
    
    image_SW = image_SW + image_data
    
    flat_image = image_data.flatten()
    counts_i, bins_i = np.histogram(flat_image, bins = int(65536/4), range = (0,65536/4))
    countsAll = countsAll + counts_i
    
image_SW = image_SW / n
    
plt.figure()
plt.imshow(image_SW, cmap = 'plasma')
plt.colorbar()
    
fig, h = plt.subplots()
h.hist(bins[:-1], bins = bins, weights = countsAll, histtype = 'step', label = "raw")
plt.show()




