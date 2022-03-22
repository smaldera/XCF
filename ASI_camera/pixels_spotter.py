from astropy.io import fits as pf
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import os


xpixels = []
ypixels = []
weight = []

files_path = ("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\acquisizione lunga\\")
background_path = ("C:\\Users\\Acer\\Tesi\\meanpedlongr.fits")


for filename in os.listdir(files_path):
    files = os.path.join(files_path, filename)
    data_f = pf.open(files, memmap = True)  
    background = pf.open(background_path, memmap=True)

    image_data = pf.getdata(files, ext=0)/4.
    back_data = pf.getdata(background_path, ext=0)

    data = image_data - back_data
    
    mask = np.where(data > 100)
    
    x, y = mask 
    w = data[mask]
    
    for i in range(0, x.shape[0]):
        xpixels.insert(i, x[i])
        ypixels.insert(i, y[i])
        weight.insert(i, w[i])


xpixels = np.array(xpixels, dtype=int)
ypixels = np.array(ypixels, dtype=int)
weight = np.array(weight, dtype=float)

#print(xpixels)
#print(weight)

%matplotlib notebook
fig, ax = plt.subplots()
ax.hist2d(xpixels, ypixels, bins = (4144, 2822), range = [[0,4144], [0,2822]], weights = weight, cmap=plt.cm.jet)

plt.figure()
plt.show()




###################### CLUSTERING ####################################################

pixels = np.stack((xpixels, ypixels), axis=0)
pixels = np.transpose(pixels)

weight = np.array(weight)


db = DBSCAN(eps=3, min_samples=2, n_jobs=1, algorithm='ball_tree').fit(pixels)
labels = db.labels_
print(labels)

unique_labels = set(labels)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


for clu_id in unique_labels:
      
    print ("CLUSTER_ID =",clu_id)
    if clu_id == -1:
        continue
      
    clu_mask = np.where(labels == clu_id)
    clu_coords = pixels[clu_mask]
    clu_weights = weight[clu_mask]

    print("cluster coord=",clu_coords)
    print("cluster weights=",clu_weights)
    
     

