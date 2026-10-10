import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import sys
sys.path.insert(0, '../../libs')
import read_sdd as sdd
import fit_histogram as fitSimo
from scipy.optimize import curve_fit

mca_file='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/SDD/Fe_14Oct2022_5mm.mca'
data_array, deadTime, livetime, fast_counts =sdd.pharse_mca(mca_file)
print("livetime=",livetime,"counts=", fast_counts, "RATE=",fast_counts/livetime,' Hz' )
print("deadTime=",deadTime)



#plot
size=len(data_array)      
bin_edges=np.linspace(0,size+1,size+1)

fig1=plt.figure(1)
plt.hist(bin_edges[:-1],bins=bin_edges,weights=data_array, histtype='step')

# fit 1st peak 
initial_pars=[22000,3950,20]
#popt1,pcov1=fitSimo.fit_Gaushistogram(data_array, bin_edges,3920,3980,initial_pars)
popt1,pcov1,xmin1,xmax1, redChi1=fitSimo.fit_Gaushistogram_iterative(data_array, bin_edges,3920,3980,initial_pars)    

print("popt1=",popt1)
print("pcov1=",pcov1)
print('recuced Chi2=',redChi1)
#plot fitted function
x=np.linspace(xmin1,xmax1,1000)
y= fitSimo.gaussian_model(x,popt1[0],popt1[1],popt1[2])
#y= fitSimo.gaussian_model(x, initial_pars[0],initial_pars [1],initial_pars[2])
plt.plot(x,y,'r-',label='fitted function')



# fit 2nd peak 
initial_pars=[8000,4350,20]
xmin=4280
xmax=4380

popt2,pcov2, xmin2,xmax2,redChi2 =fitSimo.fit_Gaushistogram_iterative(data_array, bin_edges,xmin,xmax,initial_pars)
print("popt2=",popt2)
print("pcov2=",pcov2)
print('recuced Chi2=',redChi2)
#plot fitted function
x=np.linspace(xmin2,xmax2,1000)
y= fitSimo.gaussian_model(x,popt2[0],popt2[1],popt2[2])
#y= fitSimo.gaussian_model(x, initial_pars[0],initial_pars [1],initial_pars[2])
plt.plot(x,y,'r-',label='fitted function')



################3
calib_data_x=[popt1[1],popt2[1]]
calib_data_y=[5.898,6.490]
fig1=plt.figure(2)
plt.plot(calib_data_x,calib_data_y,'ro')

poptCal, pcovCal = curve_fit(fitSimo.linear_model, calib_data_x, calib_data_y)
x=np.linspace(0,8000,8000)
y= fitSimo.linear_model(x,poptCal[0],poptCal[1])
print ("cal parameters=",poptCal)
plt.plot(x,y,'b-')

# plot calibrated spectrum:
fig3=plt.figure(3)
plt.title='calibrated spectrum'
bin_edges_cal=bin_edges*poptCal[1]+poptCal[0]
plt.hist(bin_edges_cal[:-1],bins=bin_edges_cal,weights=data_array, histtype='step')


plt.show()



