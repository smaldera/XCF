from astropy.io import fits as pf
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import os

%matplotlib notebook

files_path = ("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\acquisizione lunga\\")
background_path = ("C:\\Users\\Acer\\Tesi\\meanpedlongr.fits")

a = np.zeros(1)
b1, binsedges = np.histogram(a, bins=int(65536/4), range=(0,65536/4))

for filename in os.listdir(files_path):
    files = os.path.join(files_path, filename)
    data_f = pf.open(files, memmap = True)  
    background = pf.open(background_path, memmap=True)

    image_data = pf.getdata(files, ext=0)/4.
    back_data = pf.getdata(background_path, ext=0)

    data = image_data - back_data
    
    mask = np.where(data > 100)
    w = data[mask]
    mask = np.transpose(mask)
    
    if  mask.shape == (0, 2):
        print('EMPTY')
        continue
    
    else:
    
        db = DBSCAN(eps=3, min_samples=2, n_jobs=1, algorithm='ball_tree').fit(mask)
        labels = db.labels_
        print(labels)

        unique_labels = set(labels)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)


        for clu_id in unique_labels:
      
            print ("CLUSTER_ID =",clu_id)
            if clu_id != -1:
      
                clu_mask = np.where(labels == clu_id)
                clu_coords = mask[clu_mask]
                clu_weights = w[clu_mask]

                print("cluster coord=",clu_coords)
                print("cluster weights=",clu_weights)
            
                somma_pesi = sum(clu_weights)
                
                x_coords = list(clu_coords[:,0])
                y_coords = list(clu_coords[:,1])
  
                res = x_coords.count(x_coords[0]) == len(x_coords)
                tes = y_coords.count(y_coords[0]) == len(y_coords)
      
                if(res or tes) and len(x_coords) > 2 or len(x_coords) > 5:
                    plt.scatter(np.array(x_coords), np.array(y_coords), c = np.log(clu_weights), marker = 'x', vmin = 0, vmax = 10)
                    
                else:
                    plt.scatter(np.array(x_coords), np.array(y_coords), c = np.log(clu_weights), vmin = 0, vmax = 10)
                    c = np.histogram(somma_pesi, bins=binsedges, range=(0,65536/4))
                    b1 = b1 + c[0]
                    
    
    mask = list(mask).clear()
    mask = np.array(mask)
     

fig, ax = plt.subplots()
ax.hist(binsedges[:-1], bins = binsedges, range=(0,65536/4), weights=b1, alpha=1, histtype='step')
mean = c[0].mean()
rms = c[0].std()
s='mean='+str(round(mean,3))+"\n"+"RMS="+str(round(rms,3))
ax.text(0.7, 0.9, s,  transform=ax.transAxes,  bbox=dict(alpha=0.7))


plt.yscale("log")
plt.colorbar()
plt.show()

    
