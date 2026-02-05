import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

import xraydb as xrdb

rho_si=2.330E+00 # g/cm3



def interpolate_mu(myE, plot=False):

    #https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z14.html
    energy=np.array([  1.00000E-03, 1.50000E-03,1.83890E-03, 1.838901E-03, 2.00000E-03,3.00000E-03,4.00000E-03, 5.00000E-03, 6.00000E-03, 8.00000E-03, 1.00000E-02, 1.50000E-02, 2.00000E-02,3.00000E-02,4.00000E-02,5.00000E-02,6.00000E-02, 8.00000E-02,1.00000E-01,1.50000E-01,2.00000E-01,3.00000E-01,4.00000E-01,5.00000E-01,6.00000E-01,8.00000E-01,1.00000E+00,1.25000E+00,1.50000E+00,2.00000E+00,3.00000E+00,4.00000E+00,5.00000E+00,6.00000E+00, 8.00000E+00, 1.00000E+01, 1.50000E+01, 2.00000E+01 ])

    #mu/pho silicio!!!
    mu_su_rho=np.array([1.570E+03,5.355E+02, 3.092E+02,3.192E+03, 2.777E+03,9.784E+02,4.529E+02,2.450E+02,1.470E+02,6.468E+01,3.389E+01,1.034E+01,4.464E+00,1.436E+00,7.012E-01,4.385E-01,3.207E-01,2.228E-01,1.835E-01,1.448E-01,1.275E-01,1.082E-01,9.614E-02,8.748E-02,8.077E-02,7.082E-02,6.361E-02,5.688E-02,5.183E-02,4.480E-02,3.678E-02,3.240E-02,2.967E-02,2.788E-02,2.574E-02,2.462E-02,2.352E-02,2.338E-02 ])


    f = interpolate.interp1d(energy, mu_su_rho, kind='linear')
   #f = UnivariateSpline(energy, mu_su_rho)
    #f = interpolate.interp1d(energy, mu_su_rho, kind='cubic' )

    # get mu/rho @ myE
    #myE=2.7
    my_mu=f(myE)

    #plotting
    if plot==True:    
        x_f=np.arange(0.001,0.1,0.0001)
        y_f=f(x_f)
        
        plt.xlabel('E [keV]')
        plt.ylabel('mu/rho')
        #plt.plot(np.log10(x_f),np.log10(y_f), 'ob')
        plt.plot(x_f,y_f, 'ob')
        
        #plt.plot(np.log10(energy),np.log10(mu_su_rho),'-r')
     #   plt.plot(energy,mu_su_rho,'-r')
       
        mu_xrdb=xrdb.mu_elam('Si', x_f*1e6) # energy in eV
        plt.plot(x_f,mu_xrdb,'-g',label='Si')
        mu_xrdbBe=xrdb.mu_elam('Be', x_f*1e6) # energy in eV
        plt.plot(x_f,mu_xrdbBe,'-k',label='Be')

 #   print('E=',myE,' ===>>> mu/rho=',my_mu)

    return my_mu    




def attenuation(E,d,rho,elem='Si'):

     x=d*rho
     mu=mu_xrdb=xrdb.mu_elam(elem, E) # E in eV!!!

     att=np.exp(-mu*x)
     return att
    






 
#for name, line in xrdb.xray_lines('Mo').items():
#    print(name, ' = ', line)

#interpolate_mu(0.01, plot=True)


energy=2.3 #KeV
d=np.arange(1e-7,1e-3,1e-8) # cm
eff=1-attenuation(energy*1000.,d,rho_si,elem='Si')
plt.plot(d*1e4 ,eff,'-g',label='Si efficiency vs d E=2.3KeV')

energy2=6 #KeV
eff2=1-attenuation(energy2*1000.,d,rho_si,elem='Si')
plt.plot(d*1e4 ,eff2,'-r',label='Si efficiency vs d E=6KeV')

plt.xlabel('spessore [um]')
plt.ylabel('efficiency [1-I/I0]')

plt.legend()


plt.figure(2)
plt.plot(d*1e4,eff/eff2)
plt.xlabel('spessore [um]')
plt.ylabel('eff(2.3keV)/eff(6keV)')



plt.show()
