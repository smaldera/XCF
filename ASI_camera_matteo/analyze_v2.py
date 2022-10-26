import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import *
import glob
import sys
sys.path.insert(0, '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/libs')
import utils as al
import read_sdd as sdd
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
mySigmaMask = np.where((rms_ped > 3))

#n_files = len(f)
n_files = 5 #just for test
print("Files to be analyzed: " + str(n_files))
print("\n")

n_saved_files = 0

#np array vuoti a cui appendo..
x_all = np.empty(0)
y_all = np.empty(0)

x_allClu = np.empty(0)
y_allClu = np.empty(0)

x_cg = np.empty(0)
y_cg = np.empty(0)

n = 0.

print("--------------------------------------------------------------------------------------------------------")
print("ANALYSIS OF FIT FILES")
print("\n")

for image_file in f:
  #  print(n," --> ", image_file)
    if n % 10 == 0:
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
    
    if(n == n_files):
        break
        

print("\n\n\n\n")




###########
# plot immagine
#fig2, ax2 = plt.subplots()
#plt.plot(x_all, y_all, 'sr', alpha = 0.3, ms = 10)
#plt.plot(x_allClu, y_allClu, 'sg', alpha = 1, markerfacecolor = 'none', ms = 11)


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


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

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


# plot spettro
fig1, h1 = plt.subplots()
h1.hist(bins[:-1], bins = bins, weights = countsAll/ np.max(countsAll[np.where(np.array(bins[:-1]) > 200)]), histtype = 'step', label = "raw")
h1.hist(bins[:-1], bins = bins, weights = countsAllClu / np.max(countsAllClu), histtype = 'step', label = 'CLUSTERING')
h1.hist(bins[:-1], bins = bins, weights = countsAllOnes / np.max(countsAllOnes), histtype = 'step', label = 'Just Ones')
plt.legend()
plt.xlabel('Bins - [ADU]')
plt.ylabel('Counts - #') #è normalizzato

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

fig3, h3 = plt.subplots()
countsAllcg = countsAllcg.T
plt.imshow(countsAllcg, interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()
plt.title("Centre of gravity")

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

#GAUSSIANS ON FIRST TWO PEAKS
print("--------------------------------------------------------------------------------------------------------")
print("GAUSSIAN CURVES ON THE FRIST TWO PEAKS")
print("\n")

xdata = []  #data of x-axis
ydata = []  #data of y-axis

mean = np.empty(0)  #array for means
mean = np.append(mean, 0)   #a mean[0] ci metto zero perche voglio usare l'array da 1 in poi giusto per non fare confusione

sigma = np.empty(0) #array of sigmas
sigma = np.append(sigma, 0) #same as before



bins = bins[:-1]    #sovrascrivo i bins togliendone uno altrimenti le dimensioni di bins non concidono con i conteggi
xdata.append(bins)  #start to fill xdata

ydata.append(countsAllOnes) #start to fill ydata
ydata.append(countsAllOnes[np.where((np.array(bins) > 1800) & (np.array(bins) < 1880))])    #all counts of first peak
ydata.append(countsAllOnes[np.where((np.array(bins) > 1980) & (np.array(bins) < 2080))])    #all counts of second peak

xdata.append(bins[np.where((np.array(bins) > 1800) & (np.array(bins) < 1880))]) #all bins of first peak
xdata.append(bins[np.where((np.array(bins) > 1980) & (np.array(bins) < 2080))]) #all bins of second pea

mean = np.append(mean, np.mean(xdata[1]))   #mean for the first set of data
sigma = np.append(sigma, np.std(xdata[1]))  #sigma for the first set of data

mean = np.append(mean, np.mean(xdata[2]))   #mean for the second set of data
sigma = np.append(sigma, np.std(xdata[2]))  #sigma for the second set of data



popt, pcov = curve_fit(al.gaus, xdata[1], ydata[1], p0 = [np.max(ydata[1]), mean[1], sigma[1]]) #compute the first set of parameters for the first gaussian curve
first_peak = popt
first_peak_error = np.sqrt(np.diag(pcov))
print("Parameters for the peak k alpha of Fe55: ", first_peak)
print("Error on parameters: ", first_peak_error)
print("\n")


fig4, h4 = plt.subplots()
plt.plot(xdata[1], al.gaus(xdata[1], first_peak[0], first_peak[1], first_peak[2]), first_peak[0], first_peak[1], first_peak[2], color = 'r', label = 'Primo picco')

popt, pcov = curve_fit(al.gaus, xdata[2], ydata[2], p0 = [np.max(ydata[2]), mean[2], sigma[2]]) #compute the second set of parameters for the second gaussian curve
second_peak = popt
second_peak_error = np.sqrt(np.diag(pcov))
print("Parameters for the peak k beta of Fe55: ", second_peak)
print("Error on parameters: ", second_peak_error)
print("\n")

h4.hist(bins, bins = bins, weights = countsAllOnes, histtype = 'step')
plt.plot(xdata[2], al.gaus(xdata[2], second_peak[0], second_peak[1], second_peak[2]), second_peak[0], second_peak[1], second_peak[2], color = 'g', label = 'Secondo picco')
plt.xlabel('Bins - [ADU]')
plt.ylabel('Counts - #')

print("\n\n\n\n")

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

#LINEAR FUNCTION FOR CALIBRATION
print("--------------------------------------------------------------------------------------------------------")
print("LINEAR FUNCTION FOR CALIBRATION")
print("\n\n")

real_value = np.array([5898.75, 6490.45])   #valori k_alpha e k_beta dalla letteratura
mean  = np.delete(mean, 0, 0)   #elimino quello che era zero
popt, pcov = curve_fit(al.retta, mean, real_value)  #calcolo i parametri di best fit
first_cal = popt
first_cal_err = np.sqrt(np.diag(pcov))
print("Parameters for calibration: ", first_cal)
print("Error of parameters: ", first_cal_err)
print("\n")

x = np.linspace(0, 3000, 10)

#fig5, pic5 = plt.subplots() #disegniamo retta + scatter plot
#plt.plot(x, al.retta(x, first_cal[0], first_cal[1]), color = 'r')
#plt.scatter(mean, real_value, s = 5)

print("\n\n\n\n")


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

#ANALISYS ESCAPE PEAK
print("--------------------------------------------------------------------------------------------------------")
print("HISTOGRAM IN ENERGY + ANALISYS ESCAPE PEAK.")
print("\n\n")
#vediamo a che energie sta l'escape peak tramite una gaussiana


energy_bins = (bins * first_cal[0]) + first_cal[1]
esc_range = energy_bins [(energy_bins > 3800) & (energy_bins < 4400)]
esc_counts = countsAllOnes [(energy_bins > 3800) & (energy_bins < 4400)]

popt, pcov = curve_fit(al.gaus, esc_range, esc_counts, p0 = [np.max(esc_counts), np.mean(esc_range), np.std(esc_range)]) #compute the set of parameters for the escape peak with x-axis in energy
print("Parameters for escape peak in energy: ", popt)

esc_peak = popt
esc_peak_error = np.sqrt(np.diag(pcov))
print("Error of parameters: ", esc_peak_error)
print("\n")

print("Z test = ", np.absolute(esc_peak[1] - (5898.75 - 1740)) / esc_peak_error[1])

print("Escape peak from data = ", esc_peak[1])
print("Escape peak from literature = ", 5898.75 - 1740)
print("\n\n")


#ora vediamo a che punto sta l'escape peak e aggiungiamolo alla calibrazione
esc_range = bins [(bins > 1200) & (bins < 1350)]
esc_counts = countsAllOnes [(bins > 1200) & (bins < 1350)]
popt, pcov = curve_fit(al.gaus, esc_range, esc_counts, p0 = [np.max(esc_counts), np.mean(esc_range), np.std(esc_range)]) #compute the set of parameters for the escape peak
print("Parameters escape peak for calibration: ", popt)
print("\n")
esc_peak = popt
esc_peak_error = np.sqrt(np.diag(pcov))


mean = np.append(mean, popt[1]) #aggiungiamo al vettore chhe contiene le medie il valore dell'escape peak
real_value = np.append(real_value, 5898.75 - 1740)

popt, pcov = curve_fit(al.retta, mean, real_value)
print("Parameters linear regression calibration without errros: ", popt)
print("\n")

x = np.linspace(0, 3000, 10)

fig6, pic6 = plt.subplots()
plt.plot(x, al.retta(x, popt[0], popt[1]), color = 'r')

#questo procedimento tiene conto anche degli errori dei punti per calcolarsi i parametri
#i parametri infatti sono leggermente diversi ma in maniera 'trascurabile'
linear_model = Model(al.linear_func)
err_x = np.array([first_peak_error[1], second_peak_error[1], esc_peak_error[1]])
data = RealData(mean, real_value, err_x)
odr = ODR(data, linear_model, beta0 = [3., 70.])
print("Parameters linear regression for calibration with errors: ")
out = odr.run()
out.pprint()
second_cal = out.beta
second_cal_err = out.sd_beta
print("\n\n")
plt.errorbar(mean, real_value, xerr = err_x, fmt = '.')
plt.plot(x, al.retta(x, second_cal[0], second_cal[1]), color = 'b', linewidth = 0.5)
plt.title('CMOS Calibration')
plt.xlabel('Channels - [ADU]')
plt.ylabel('Energy - [eV]')
print("--------------------------------------------------------------------------------------------------------")

#DA CHIEDERE SE EFFETTIVAMENTE VA BENE QUESTO SECONDO APPROCCIO CHE TIENE CONTO ANCHE DEGLI ERRORI


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

#SDD CALIBRATION
print("SDD CALIBRATION")
print("\n\n")

data_array, deadTime, livetime, fast_counts = sdd.pharse_mca('/Volumes/FILMS/dati_ASI294/misure_collimatore_14Oct/SDD/Fe_14Oct2022_5mm.mca')
size = len(data_array)
bins_edges = np.linspace(0, size + 1, size + 1)

sdd_xdata = []  #data of x-axis
sdd_ydata = []  #data of y-axis

sdd_mean = np.empty(0)  #array for means
sdd_mean = np.append(sdd_mean, 0)   #a mean[0] ci metto zero perche voglio usare l'array da 1 in poi giusto per non fare confusione

sdd_sigma = np.empty(0) #array of sigmas
sdd_sigma = np.append(sdd_sigma, 0) #same as before

bins_edges = bins_edges[:-1]    #sovrascrivo i bins togliendone uno altrimenti le dimensioni di bins non concidono con i conteggi
sdd_xdata.append(bins_edges)  #start to fill xdata

sdd_ydata.append(data_array) #start to fill ydata
sdd_ydata.append(data_array[np.where((np.array(bins_edges) > 3790) & (np.array(bins_edges) < 4070))])    #all counts of first peak
sdd_ydata.append(data_array[np.where((np.array(bins_edges) > 4240) & (np.array(bins_edges) < 4430))])    #all counts of second peak
sdd_ydata.append(data_array[np.where((np.array(bins_edges) > 2640) & (np.array(bins_edges) < 2890))])    #all counts of escape peak

sdd_xdata.append(bins_edges[np.where((np.array(bins_edges) > 3790) & (np.array(bins_edges) < 4070))]) #all bins of first peak
sdd_xdata.append(bins_edges[np.where((np.array(bins_edges) > 4240) & (np.array(bins_edges) < 4430))]) #all bins of second peak
sdd_xdata.append(bins_edges[np.where((np.array(bins_edges) > 2640) & (np.array(bins_edges) < 2890))]) #all bins of escape peak

sdd_mean = np.append(sdd_mean, np.mean(sdd_xdata[1]))   #mean for the first set of data
sdd_sigma = np.append(sdd_sigma, np.std(sdd_xdata[1]))  #sigma for the first set of data

sdd_mean = np.append(sdd_mean, np.mean(sdd_xdata[2]))   #mean for the second set of data
sdd_sigma = np.append(sdd_sigma, np.std(sdd_xdata[2]))  #sigma for the second set of data

sdd_mean = np.append(sdd_mean, np.mean(sdd_xdata[3]))   #mean for the second set of data
sdd_sigma = np.append(sdd_sigma, np.std(sdd_xdata[3]))  #sigma for the second set of data

popt, pcov = curve_fit(al.gaus, sdd_xdata[1], sdd_ydata[1], p0 = [np.max(sdd_ydata[1]), sdd_mean[1], sdd_sigma[1]]) #compute the first set of parameters for the first gaussian curve
sdd_first_peak = popt
sdd_first_peak_error = np.sqrt(np.diag(pcov))
print("Parameters for the peak k alpha of Fe55 with SDD: ", sdd_first_peak)
print("Error on parameters: ", sdd_first_peak_error)
print("\n")

fig7, h7 = plt.subplots()

plt.plot(sdd_xdata[1], al.gaus(sdd_xdata[1], sdd_first_peak[0], sdd_first_peak[1], sdd_first_peak[2]), sdd_first_peak[0], sdd_first_peak[1], sdd_first_peak[2], color = 'r', label = 'Primo picco')

popt, pcov = curve_fit(al.gaus, sdd_xdata[2], sdd_ydata[2], p0 = [np.max(sdd_ydata[2]), sdd_mean[2], sdd_sigma[2]]) #compute the second set of parameters for the second gaussian curve
sdd_second_peak = popt
sdd_second_peak_error = np.sqrt(np.diag(pcov))
print("Parameters for the peak k beta of Fe55 with SDD: ", sdd_second_peak)
print("Error on parameters: ", sdd_second_peak_error)
print("\n")

plt.plot(sdd_xdata[2], al.gaus(sdd_xdata[2], sdd_second_peak[0], sdd_second_peak[1], sdd_second_peak[2]), sdd_second_peak[0], sdd_second_peak[1], sdd_second_peak[2], color = 'g', label = 'Secondo picco')


popt, pcov = curve_fit(al.gaus, sdd_xdata[3], sdd_ydata[3], p0 = [np.max(sdd_ydata[3]), sdd_mean[3], sdd_sigma[3]]) #compute the third set of parameters for the third gaussian curve
sdd_escape_peak = popt
sdd_escape_peak_error = np.sqrt(np.diag(pcov))
print("Parameters for the escape peak of Fe55 in silicon with SDD: ", sdd_escape_peak)
print("Error on parameters: ", sdd_escape_peak_error)
print("\n")

h7.hist(bins_edges, bins = bins_edges, weights = data_array, histtype = 'step') #SDD histogram
plt.plot(sdd_xdata[3], al.gaus(sdd_xdata[3], sdd_escape_peak[0], sdd_escape_peak[1], sdd_escape_peak[2]), sdd_escape_peak[0], sdd_escape_peak[1], sdd_escape_peak[2], color = 'y', label = 'Escape peak')
plt.title('SDD Gaussians on histogram')
plt.xlabel('Bins - [ADU]')
plt.ylabel('Counts - #')
plt.xlim([0, 7.5e3])


fig8, pic8 = plt.subplots()

linear_model = Model(al.linear_func)
err_x = np.array([sdd_first_peak_error[1], sdd_second_peak_error[1], sdd_escape_peak_error[1]])
sdd_mean  = np.delete(sdd_mean, 0, 0)   #elimino quello che era zero
data = RealData(sdd_mean, real_value, err_x)
odr = ODR(data, linear_model, beta0 = [3., 70.])
print("Parameters linear regression for SDD calibration with errors: ")
out = odr.run()
out.pprint()
sdd_cal = out.beta
sdd_cal_err = out.sd_beta
print("\n\n")
x = np.linspace(0, 4450, 10)
plt.plot(x, al.retta(x, sdd_cal[0], sdd_cal[1]), color = 'r', linewidth = 0.7)
plt.title('SDD Calibration')
plt.xlabel('Channels - [ADU]')
plt.ylabel('Energy - [eV]')
plt.errorbar(sdd_mean, real_value, xerr = err_x, fmt = '.')

print("--------------------------------------------------------------------------------------------------------")
print("\n\n\n\n")

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

#HISTOGRAM IN ENERGY + SDD
print("HISTOGRAM IN ENERGY + SDD")
print("\n\n")

energy_bins = (bins * second_cal[0]) + second_cal[1]
sdd_energy_bins = (bins_edges * sdd_cal[0]) + sdd_cal[1]

fig22, h22 = plt.subplots()

h22.hist(sdd_energy_bins, bins = sdd_energy_bins, weights = data_array / np.max(data_array), histtype = 'step', label = 'SDD')
h22.hist(energy_bins, bins = energy_bins, weights = countsAllOnes / np.max(countsAllOnes), histtype = 'step', label = 'CMOS')
plt.legend()
plt.title('Energy histograms CMOS + SDD')
plt.xlim([0, 1.e4])
plt.xlabel('[eV]')
plt.ylabel('#')


# save histos
#np.savez(shots_path+'spectrum_all_raw', counts = countsAll, bins = bins)
#np.savez(shots_path+'spectrum_allCLU_cut25', counts = countsAllClu, bins = bins)



plt.show()

