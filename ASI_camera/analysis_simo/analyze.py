import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT
from pedestal import bg_map


def retrive_vectors(nomefile):
    data=np.load(nomefile)
    w=data['w']
    x_pix=data['x_pix']
    y_pix=data['y_pix']

    return w,x_pix,y_pix

def retrive_histo(nomefile):
    data=np.load(nomefile)
    counts=data['counts']
    bins=data['bins']
    fig, ax = plt.subplots()
    #print ("len(coutsAll)=",len(countsAll) )
    ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step')
    plt.show()







shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/Fe55/source/'
bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/Fe55/bkg/'
create_bg_map=False
outRootfile_name=shots_path+'histo_all.root'

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
outRootfile=ROOT.TFile(outRootfile_name,'recreate')
h1=ROOT.TH1F('h1','',16384,0,16384)

#aa=[[0, 0]]

# array vuoti a cui appendo 
supp_weightsAll=np.empty(0)
x_pix=np.empty(0)
y_pix=np.empty(0)

n=0
#print("shape iniziale=",supp_coordsAll.shape)
#print ("supp_coordsAll inizila=",supp_coordsAll)

zero_img=np.zeros((2822, 4144))
n_saved_files=0
for image_file in f:

  #  print('===============>>>  n=',n)
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
    zero_img=zero_img+image_data

    supp_coords_i, supp_weigths_i= al.select_pixels2(image_data,60)
#    print(' supp_coords_i=', supp_coords_i, " supp_weigths_i=",supp_weigths_i )
    print(' supp_coords_i=', supp_coords_i )
    

    traspose=np.transpose(supp_coords_i)
    x_pix=np.append(x_pix,traspose[0])
    y_pix=np.append(y_pix,traspose[1])
    print ('x_pix=',x_pix)
    print ('y_pix=',y_pix)

    # ricreo supp_coords:
    supp_coords2=np.empty(0)
   # for i in range(0,len(x_pix)):
   #     a=[x_pix[i],y_pix[i] ]
   #     aa=np.array(a)
   #     supp_coords2=np.append(  supp_coords2,aa)

    supp_coords2=np.append(  supp_coords2,x_pix)
    supp_coords2=np.append(  supp_coords2,y_pix)
    supp_coords2= supp_coords2.reshape(2,len(x_pix))
    supp_coords2=np.transpose(supp_coords2)
    print("supp coords ricreato=",supp_coords2)    

    
    supp_weightsAll=np.append( supp_weightsAll, supp_weigths_i)
    
    break 
    
    if n%100==0 and n>0:
        n_saved_files+=1
        print('saving '+str(n)+' events, n_file=',str(n_saved_files))
        out_file=shots_path+'shots_'+str(n_saved_files)
        np.savez(out_file,w=supp_weightsAll, x_pix=x_pix, y_pix=y_pix)
        break
    n=n+1


#h1.Draw()
#input("press key to continue")
al.plot_image(zero_img)
outRootfile.cd()
h1.Write()
outRootfile.Close()
#h1.Draw()
input("press key to continue")


print ('retriving vectors...')
w,x,y=retrive_vectors(shots_path+'shots_1.npz')

print('w=',w)
print('x=',x)
print('y=',y)


plt.scatter(x,y,c = np.log10(w) )
plt.colorbar()
plt.show()

