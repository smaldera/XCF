from astropy.io import fits as pf
import numpy as np
from matplotlib import pyplot as plt

def isto_all(image_data):
    flat_image=image_data.flatten()
    fig, ax = plt.subplots()
    ax.hist(flat_image, bins=int(65536/4), range=(0,65536/4)   , alpha=1, histtype='step')
    mean=flat_image.mean()
    rms=flat_image.std()
    s='mean='+str(round(mean,3))+"\n"+"RMS="+str(round(rms,3))
    ax.text(0.7, 0.9, s,  transform=ax.transAxes,  bbox=dict(alpha=0.7))
    plt.show()

def plot_image(image_data):
    plt.figure()
    plt.imshow(image_data, cmap='plasma')
    plt.colorbar()
    plt.show()
    
def read_image(nomefile): # REMEMBER - you need to divide by 4 (see ADC)
    pf.open(nomefile, memmap=False)
    image_data = pf.getdata(nomefile, ext=0)
    return image_data

def retrive_vectors_old(nomefile):
    data=np.load(nomefile)
    w=data['w']
    x_pix=data['x_pix']
    y_pix=data['y_pix']
    return w,x_pix,y_pix

def retrive_vectors(nomefile):
    data=np.load(nomefile)
    w=data['w']
    x_pix=data['x_pix']
    y_pix=data['y_pix']
    size=data['sizes']
    return w, x_pix, y_pix, size

# DEPRECATED!!!!!
def retrive_histo(nomefile):
    data=np.load(nomefile)
    counts=data['counts']
    bins=data['bins']
    fig, ax = plt.subplots()
    histo=ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step')
    return histo 

def save_histo(outHisto_name,countsAll,bins):
    np.savez(outHisto_name,counts=countsAll,bins=bins)

def save_vectors(out_file, supp_weightsAll,x_pix,y_pix):
    np.savez(out_file,w=supp_weightsAll, x_pix=x_pix, y_pix=y_pix)

def save_vectors2(out_file, supp_weightsAll,x_pix,y_pix,n_img):
    np.savez(out_file,w=supp_weightsAll, x_pix=x_pix, y_pix=y_pix,n_img=n_img)

def select_pixels(image_data, threshold=100, upper=100000):
    mask_zeroSupp=np.where( (image_data>threshold) &( image_data<upper) )
    supp_coords=mask_zeroSupp
    supp_weights=image_data[mask_zeroSupp]
    return supp_coords, supp_weights

def select_pixels_RMS(image_data, rms_ped, nSigma=5., upper=100000): #  select pixele above n sigma 
    image_selection=image_data-(nSigma*rms_ped)
    mask_zeroSupp=np.where( (image_selection>0.) & ( image_data<upper) )
    supp_coords=mask_zeroSupp
    supp_weights=image_data[mask_zeroSupp]
    return supp_coords, supp_weights
  
def write_fitsImage(array, nomefile,overwrite='False' ):
    hdu = pf.PrimaryHDU(array)
    hdu.writeto(nomefile,overwrite=overwrite)