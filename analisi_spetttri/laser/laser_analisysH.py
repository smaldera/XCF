import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../libs')
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
import utils_v2 as al
import os
from scipy.optimize import curve_fit

# range nel qie cercare il picco da fittare
def fit_peak(filename, fileFormat='npz',min_range=1, max_range=100000):

   p=histogramSimo()
   p.read_from_file(filename, fileFormat )

 
   #cerco x del massimo in un certo range:
   bin_centers=fitSimo.get_centers(p.bins)
   mask=np.where((bin_centers>min_range)&(bin_centers<max_range))
   c2=p.counts[mask]
   h=np.max(c2)
   xmax=bin_centers[p.counts==h]
   print('xmax=',xmax,' h=',h)

   # qua fa il fit...
   popt,  pcov, xmin,xmax, redChi2= fitSimo.fit_Gaushistogram_iterative(p.counts,p.bins,xmin=min_range,xmax=max_range, initial_pars=[h,xmax,10], nSigma=1.5 )

   print('mean=',popt[1], ' +-',pcov[1][1]**0.5)
   print('sigma=',popt[2], ' +-',pcov[2][2]**0.5)
   print('N=',popt[0], ' +-',pcov[0][0]**0.5)
   print('CHI2/NDoF= ',redChi2)

   return  p,popt,  pcov, xmin,xmax, redChi2





XBINS=2822
YBINS=4144

commonpath='/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/6giu/'
nameXdist='X_dist_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz'
nameYdist='Y_dist_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz'
name_image='imageRaw_pixCut15.0sigma.fits'
name_spectrum='spectrum_all_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz'

x_histos=['h_6.3_p10000/X_dist_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz']
y_histos=['h_6.3_p10000/Y_dist_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz']
images=['h_6.3_p10000/imageRaw_pixCut15.0sigma.fits']
spectrum=['h_6.3_p10000/spectrum_all_ZeroSupp_pixCut15.0sigma_CLUcut_150.0sigma.npz']

#dist=[3.3,4.3,5.3,6.3,7.3,7.65,8,9,10,11,12]
#folders=['h_3.3/','h_4.3/','h_5.3/','h_6.3/','h_7.3/','h_7.65/','h_8/','h_9/','h_10/','h_11/','h_12/']
#dist=[3.3,4.3,5.3,]
#folders=['h_3.3/','h_4.3/','h_5.3/']



dist=[3.3,4.3,5.3,6.3,7.3,7.65,8,9,10,11,12]
folders=['h_3.3/','h_4.3/','h_5.3/','h_6.3/','h_7.3/','h_7.65/','h_8/','h_9/','h_10/','h_11/','h_12/']

suffix='_m10000/'
outdir='plots/m10k/'
os.system('mkdir -p '+commonpath+outdir)

meanX=[]
meanXerr=[]
meanY=[]
meanYerr=[]

for i in range(0,len(dist)):

      #folders[i]=folders[i][:-1]+suffix
      print("folder=",folders[i])
      x_histoname=commonpath+folders[i]+nameXdist
      y_histoname=commonpath+folders[i]+nameYdist
      imagefileName=commonpath+folders[i]+name_image
      spectName=commonpath+folders[i]+name_spectrum
      
      pX,poptX,  pcovX, xminX,xmaxX, redChi2X= fit_peak(x_histoname)
      pY,poptY,  pcovY, xminY,xmaxY, redChi2Y= fit_peak(y_histoname)
      

      meanX.append(poptX[1])
      meanY.append(poptY[1])
      meanXerr.append( pcovX[1][1]**0.5)
      meanYerr.append( pcovY[1][1]**0.5)

      
      #draw
      fig=plt.figure(i,(18,10))
      axs = fig.subplots(2,2)
      fig.suptitle('dist='+str(dist[i]))

      axs[0,0].set_title('X dist')
      pX.plot(axs[0,0],'x_dist')
      #plt.hist(pX.bins[:-1],bins=pX.bins,weights=pX.counts, histtype='step', label='x dist')
      x=np.linspace(xminX,xmaxX,1000)
      y= fitSimo.gaussian_model(x,poptX[0],poptX[1],poptX[2])
      axs[0,0].plot(x,y,'r-')
      s ='mean=' + str(round(poptX[1], 3)) + " +- " + str(round( pcovX[1][1]**0.5 ,3))+'\n sigma='+ str(round(poptX[2], 3)) + ' +- ' + str(round( pcovX[2][2]**0.5 ,3))+'\n Chi2/Ndof='+str(round(redChi2X,3)) 
      axs[0,0].text(0.7, 0.8, s,  transform = axs[0,0].transAxes,  bbox = dict(alpha = 0.7))
      axs[0,0].set_xlim([poptX[1]-4*poptX[2], poptX[1]+4*poptX[2]  ])

      # y dist:
      axs[0,1].set_title('Y dist')
      pY.plot(axs[0,1],'Y_dist')
      x=np.linspace(xminY,xmaxY,1000)
      y= fitSimo.gaussian_model(x,poptY[0],poptY[1],poptY[2])
      axs[0,1].plot(x,y,'r-')
      s = 'mean=' + str(round(poptY[1], 3)) + " +- " + str(round( pcovY[1][1]**0.5 ,3))+'\n sigma='+ str(round(poptY[2], 3)) + " +- " + str(round( pcovY[2][2]**0.5 ,3))+'\n Chi2/Ndof='+str(round(redChi2Y,3))
      axs[0,1].text(0.7, 0.8, s,  transform = axs[0,1].transAxes,  bbox = dict(alpha = 0.7))
      axs[0,1].set_xlim([poptY[1]-4*poptY[2], poptY[1]+4*poptY[2]  ])

      
      # spectrum:
      axs[1,0].set_title('counts spectrum')
      pCounts=histogramSimo()
      pCounts.read_from_file(spectName,'npz')
      pCounts.plot(axs[1,0],'counts')

      #image
      axs[1,1].set_title('counts map')
      image_data = al.read_image(imagefileName)
      axs[1,1].imshow(image_data, origin='lower',  extent=[0, XBINS, 0, YBINS])
      axs[1,1].set_ylim([poptY[1]-4*poptY[2], poptY[1]+4*poptY[2]  ])
      axs[1,1].set_xlim([poptX[1]-4*poptX[2], poptX[1]+4*poptX[2]  ])

      fig.savefig(commonpath+outdir+'dist_'+str(dist[i])+'all.png')




dist=np.array(dist)
meanX=np.array(meanX)      
meanY=np.array(meanY)
meanXerr=np.array(meanXerr)
meanYerr=np.array(meanYerr)

#azzero:
meanX=meanX-meanX[0]
meanY=meanY-meanY[0]
dist=dist-dist[0]
sensor_dist=((meanX**2+meanY**2)**0.5)*4.6*1e-3

meanXerr=(meanXerr**2+meanXerr[0]**2)**0.5
meanYerr=(meanYerr**2+meanYerr[0]**2)**0.5

sensor_distErr= (((meanX/(( meanX**2+meanY**2 )**0.5))**2)*meanXerr**2+  ((meanY/(( meanX**2+meanY**2 )**0.5))**2)*meanYerr**2)**0.5
sensor_distErr[0]=np.mean(sensor_distErr[1:])

print("sensor_dist err=", sensor_distErr)

sensor_distErr=sensor_distErr*4.6*1e-3



fig3=plt.figure()
ax3 = fig3.subplots()
ax3.errorbar(meanX,meanY,yerr=meanYerr, xerr=meanXerr, fmt='ro'  )
poptCal, pcovCal = curve_fit(fitSimo.linear_model,meanX , meanY , absolute_sigma=True,    bounds=(-np.inf, np.inf )   )
x=np.linspace(min(meanX)-2 ,max(meanX)+2,8000)
y= fitSimo.linear_model(x,poptCal[0],poptCal[1])
ax3.plot(x,y,'k-')
p0=poptCal[0]
p1=poptCal[1]
p0Err=(pcovCal[0][0])**0.5
p1Err=(pcovCal[1][1])**0.5

print("/n FIT XY =========== ")
print('P0=',p0," +- ",p0Err)
print('P1=',p1," +- ",p1Err)






fig2=plt.figure()
ax = fig2.subplots()


#ax.plot(dist,sensor_dist,'ro')
ax.errorbar(dist,sensor_dist,yerr=sensor_distErr, fmt='ro'  )
xx= np.arange(0,10,0.1)
yy=xx
ax.plot(xx,yy,'--b')
ax.set_xlabel('delta h')
ax.set_ylabel('sensor dist.')

    
print("/n FIT distance  =========== ")
#poptCal, pcovCal = curve_fit(fitSimo.linear_model, dist, sensor_dist , absolute_sigma=True,  sigma= sensor_distErr,   bounds=(-np.inf, np.inf )   )
poptCal, pcovCal = curve_fit(fitSimo.linear_model0, dist, sensor_dist , absolute_sigma=True,  sigma= sensor_distErr,   bounds=(-np.inf, np.inf )   )

#chisq = (((dist - fitSimo.linear_model(dist,poptCal[0],poptCal[1]))/sensor_distErr)**2).sum()
chisq = (((dist - fitSimo.linear_model0(dist,poptCal[0]))/sensor_distErr)**2).sum()

ndof= len(dist) - len(poptCal)
redChi2=chisq/ndof
print('chi2=',chisq," ndof=",ndof, " chi2/ndof=",redChi2)

x=np.linspace(min(dist)-2 ,max(dist)+2,8000)
#y= fitSimo.linear_model(x,poptCal[0],poptCal[1])
y= fitSimo.linear_model0(x,poptCal[0])

ax.plot(x,y,'k-')

print ("cal parameters=",poptCal)
print("cov matrix=",pcovCal)
   

#p0=poptCal[0]
#p1=poptCal[1]
p1=poptCal[0]



#p0Err=(pcovCal[0][0])**0.5
#p1Err=(pcovCal[1][1])**0.5
#p0p1Cov=pcovCal[0][1]

p1Err=(pcovCal[0][0])**0.5


#print("p0=",p0," p0Err=",p0Err," p1=",p1,"p1Err=",p1Err," covp0p1=",p0p1Cov)
#string ='p0='+ str(round(p0, 5)) + " +- " + str(round( p0Err ,5))+'\n p1='+ str(round(p1, 5)) + ' +- ' + str(round( p1Err ,5))

print("p1=",p1,"p1Err=",p1Err)
string ='P1='+str(round(p1, 6)) + ' +- ' + str(round( p1Err ,7))


ax.text(0.2, 0.8, string,  transform = ax.transAxes,  bbox = dict(alpha = 0.7))
ax.set_title('Theta m10k steps')



#calP0=-p0/p1
#calP1=1./p1
#calP1Err=((1./p1)**2)*p1Err
#calP0Err=( ((1./p1)*p0Err)**2+  (p1Err*p0/(p1**2))**2  )**0.5
    
    


#    print("CAL P0=",calP0," calP0Err=",calP0Err,"  calP1= ",calP1, "  calP1Err=",calP1Err)
#    plt.figure(n_files+2)
#    plt.errorbar(fitted_mean,true ,xerr=fitted_meanErr, fmt='ro')
#    x=np.linspace(min(fitted_mean)-2000 ,max(fitted_mean)+2000,1000)
#    y= fitSimo.linear_model(x,calP0,calP1)
#    y_err=y+3*((x*calP1Err)**2+calP0Err**2)**0.5
#    plt.plot(x,y,'b-')
#    plt.plot(x,y_err,'k-')
     


plt.show()
