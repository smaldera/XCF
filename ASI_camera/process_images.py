import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as col
import glob
import utils as al




import ROOT

path='/home/maldera/Desktop/eXTP/ASI294/testImages/testFe/2022-02-11_11_56_05Z_src5sec/'
f=glob.glob(path+"/*.FIT")
pedfile='mean_pedLong.fits'
mean_ped=al.read_image(pedfile) # non divido per 4. il ped e' gia' stato diviso alla lettura delle immagini 

#creo un istogramma vuoto
x=[]
countsAll,bins=np.histogram(x,bins=int(65536/4)  ,range=(0,65536/4)  )
h1=ROOT.TH1F('h1','',16384,0,16384)
h2=ROOT.TH1F('h2','',16384,0,16384)

aa=[[0, 0]]
supp_coordsAll=np.empty((0,2))
supp_weightsAll=np.empty(0)

x_pix=np.empty(0)
y_pix=np.empty(0)
n=0
print("shape iniziale=",supp_coordsAll.shape)
print ("supp_coordsAll inizila=",supp_coordsAll)

sumq=[]

for image_file in f:

    print('===============>>>  n=',n)
    
    image_data=al.read_image(image_file)/4.
    # subtract bkg:
    image_data=image_data-mean_ped
    flat_image=image_data.flatten()
    mean=flat_image.mean() 
    if mean > 100:
        print("skipping image ",image_file,"  mean= ",mean)
        continue
    
    counts_i,bins_i=np.histogram(flat_image,bins=int(65536/4)  ,range=(0,65536/4)  )
    countsAll=countsAll+counts_i
    #for i in range (0, len(flat_image): )
    #    h1.Fill(flat_image[i])
    w=np.ones(len(flat_image))
    h1.FillN(len(flat_image), flat_image, w)
    #al.isto_all(image_data)
    #al.isto_all(mean_ped)


    # analisi pixels sopra soglia..
    supp_coords_i, supp_weigths_i= al.select_pixels2(image_data)
    print(' supp_coords_i=', supp_coords_i, " supp_weigths_i=",supp_weigths_i )
    traspose=np.transpose(supp_coords_i)
    x_pix=np.append(x_pix,traspose[0])
    y_pix=np.append(y_pix,traspose[1])
    supp_weightsAll=np.append( supp_weightsAll, supp_weigths_i)
    

    # clustering!!!
    if len(supp_weigths_i)>0:
       clu_q= al.clustering( supp_coords_i, supp_weigths_i)
       sumq=sumq+clu_q
       print("clu_q=",clu_q," sumq=  ",sumq  )
       
   # if n==10:
   #     break
    n=n+1
sumq=np.array(sumq)    
w2= np.ones(len(sumq))
print("type(sumq) ",type(sumq))
h2.FillN(len(sumq), sumq, w2 )
h2.Draw()

    
#plot final histo:
#fig, ax = plt.subplots()
#print ("len(coutsAll)=",len(countsAll) )
#ax.hist(bins[:-1],bins=bins,weights=countsAll, histtype='step')

#traspose=np.transpose(supp_coordsAll)
#x=traspose[0]
#y=traspose[1]

print("x=",x_pix)
print('y=',y_pix)
print('supp_weightsAll=', supp_weightsAll)
plt.figure()
#plt.hist2d(x,y, bins=[image_data.shape[0],image_data.shape[1] ],  weights=supp_weights)
#plt.hist2d(x_pix,y_pix, bins=[2822,4144], weights= supp_weightsAll , range=[[0, 2822], [0, 4144]], norm=col.LogNorm() )

#plt.plot(x_pix,y_pix,'or')
plt.scatter(x_pix,y_pix, c=supp_weightsAll)
plt.colorbar()


al.isto_all(np.array( sumq))

plt.show()    


