from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np



def interpolate_mu(myE, plot=0):

    x_energy=[-0.1,1.,1.5,2.,3.,3.2,4.,5.,6.,8.,10,20,50]  # KeV
    mu_su_rho=[5e3,3.6e3,1.191e3,5.279e2,1.625e2,1.34e2,7.778e1,3.931e1,2.27e1,9.446,5.120, 7.779e-1,2.080E-01]


    f = interpolate.interp1d(x_energy, mu_su_rho, kind='cubic' )

    # get mu/rho @ myE
    #myE=2.7
    my_mu=f(myE)

    #plotting
    if plot==1:    
        x_f=np.arange(1,4,0.1)
        y_f=f(x_f)
        plt.plot(x_energy,mu_su_rho, 'or')
        plt.xlabel('E [keV]')
        plt.ylabel('mu/rho')
        plt.plot(x_f,y_f, '-b')
        plt.axvline(x=myE, linestyle='--') 
        plt.axvline(x=myE, linestyle='--') 
        plt.axhline(y=my_mu, xmax=myE,  linestyle='--') 

        plt.show()

 #   print('E=',myE,' ===>>> mu/rho=',my_mu)

    return my_mu    




def attenuation_vs_d(E, d,  plot=0):

    
    # air attenuation
    air_rho=1.21e-3 # [gcm-3]
    my_mu= interpolate_mu(E)     
    air_att=np.exp(-(d)*my_mu*air_rho)

    if plot==1:
        plt.figure()
        plt.semilogy(E,air_att, '-b',label='air attenuation at 10cm')
        
        plt.ylabel('attenuation')
        plt.xlabel('E [keV]')
        plt.legend()

        plt.show()
    
    return air_att




if __name__ == "__main__":


    Ebins=np.arange(2,10,0.1)
    E=2.3
    d=np.arange(1,20,0.1)
    att=attenuation_vs_d(Ebins,10,1)

   # plt.figure()
   # plt.plot(d,att, '-b',label='air attenuation at 2.3 keV')
   # plt.show()
    
    print("att=",att)

