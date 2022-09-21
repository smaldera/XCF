import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT

def pulisci(w,x_pix,y_pix):

               
    w2=np.empty(0)
    x2=np.empty(0)
    y2=np.empty(0)


    for i in range(0,len(w)):

        trovato=0
        

        for j in range (0,len(w2)):
            if (w[i]==w2[j]) and(x_pix[i]==x2[j]) and(y_pix[i]==y2[j] ):
                trovato=1
                break

        if trovato==0:    
            w2=np.append(w2,w[i])
            x2=np.append(x2,x_pix[i])
            y2=np.append(y2,y_pix[i])
           
            
            

    #print('w2=',w2)
    #print('x2=',x2)
    #print('y2=',y2)

    return w2,x2,y2
    
def invert_transpose(x,y):
 
    supp_coords2=np.empty(0)
 
    supp_coords2=np.append(  supp_coords2,x)
    supp_coords2=np.append(  supp_coords2,y)
    supp_coords2= supp_coords2.reshape(2,len(x))
    supp_coords2=np.transpose(supp_coords2)
    print("supp coords ricreato=",supp_coords2)    

    return  supp_coords2



def read_histos(path):
    f=glob.glob(path+"/histoAll_reducedData*.npz")

    counts_all=np.empty(0)
    bins_all=np.empty(0)
    
    n=0
    for hist_file in f:
        print('===============>>>  file= ',hist_file)
        data=np.load(hist_file)
        counts=data['counts']
        bins=data['bins']
        if n==0:
            counts_all=np.append(counts_all,counts)
            bins_all=np.append (bins_all,bins)
        else:
            counts_all=counts_all+counts
        
        n=n+1

        mean=np.sum(counts*bins_all[:-1])/np.sum(bins_all[:-1])
        print ("n=",n," mean= ",mean)
        
    return counts_all,bins_all     
            

def read_all_files(path):
    f=glob.glob(path+"/reducedData*.npz")
    supp_weightsAll=np.empty(0)
    w_clusterAll=np.empty(0)
    x_pixAll=np.empty(0)
    y_pixAll=np.empty(0)
    n=0
    for shot_file in f:
        print('===============>>>  file= ',shot_file)
        w,x_pix,y_pix=al.retrive_vectors(shot_file)

     

        #mask=np.where( (w>1))
        supp_weightsAll=np.append( supp_weightsAll, w)
        x_pixAll=np.append( x_pixAll, x_pix)
        y_pixAll=np.append( y_pixAll, y_pix)
        n=n+1

        print ("w=", w)
        print ("x=", x_pix)
        print ("y=", y_pix)

        
    print('1111111111111111111111111111')    
    supp_weightsAll,x_pixAll,y_pixAll =pulisci(supp_weightsAll,x_pixAll,y_pixAll)  
    # provo clustering...
    if (len(x_pixAll)>1):
        coords= invert_transpose(x_pixAll,y_pixAll)
        w_clusterAll=al.clustering(coords,supp_weightsAll )
        #w_clusterAll=np.append(  w_clusterAll, w_cluster)
        
    return supp_weightsAll,x_pixAll,y_pixAll,  w_clusterAll



#####################################################

shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/misureFe_1?.7/Fe/'
bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/bkg/'
outHistoFe_name=shots_path+'histo_all.npz'

wAll,x_pixAll,y_pixAll, w_clusterAll= read_all_files(shots_path)
# mask:
mask=np.where( (y_pixAll>1250) & (y_pixAll<3000) &  (x_pixAll>1000) & (x_pixAll<1800 )) # condizioni multiple con bitwise operators okkio alla precedenza!!


#print (wAll)







#shots_pathBGlong='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/test2/CapObj/'
#wBgL,x_pixBgL,y_pixBgL= read_all_files(shots_pathBGlong)
#maskBgL=np.where( (wBgL>100)) # condizioni multiple con bitwise operators okkio alla precedenza!!


#fig, ax = plt.subplots()
counts_red,bins_red=np.histogram(wAll,bins=int(65536/4)  ,range=(0,65536/4)  )
#ax.hist(bins_red[:-1],bins=bins_red,weights=counts_red, histtype='step',label='reduced')
h_red=al.isto_all_root(counts_red)

counts_cluster,bins_clu=np.histogram(w_clusterAll,bins=int(65536/4.)  ,range=(0,65536/4)  )
#ax.hist(bins_clu[:-1],bins=bins_clu,weights=counts_cluster, histtype='step',label='cluster')


          
histoAll_path=shots_path
counts, bins = read_histos(shots_path)
#data=np.load( '/home/maldera/datiFe_12.7/histoAll_reducedData_15.npz')
#counts=data['counts']
#bins=data['bins']

#ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step',label='all')
#plt.legend()
#plt.show()




    
h_clu=ROOT.TH1F('h_clu','',16384,0,16384)
c=np.float64(w_clusterAll) 
w=np.ones(len(c))
h_clu.FillN(len(c), c, w)

h_red=ROOT.TH1F('h_red','',16384,0,16384)
c2=np.float64(wAll) 
w=np.ones(len(c2))
h_red.FillN(len(c2), c2, w)
h_red.SetLineColor(2)


c=ROOT.TCanvas('c1','',0)
#h_clu.Draw('same')
#h_red.Draw()
h_clu.Draw('')

leg=ROOT.TLegend(0.6,0.7,0.9,0.9)
leg.AddEntry(h_red, 'reduced data','l')
leg.AddEntry(h_clu, 'clustering','l')

#leg.Draw()

#######################
#  scatter plot

#fig=plt.figure(figsize=(10,8))
#fig.subplots_adjust(left=0.085, right=0.97, top=0.95, bottom=0.09,hspace=0.34,wspace=0.255)
#ax=plt.subplot(111)
#plt.scatter(x_pixAll,y_pixAll,c = np.log10(wAll) )
#plt.colorbar()
#plt.title('Fe55 source')

plt.show()
