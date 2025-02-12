from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np
import math
import glob
import pandas
import datetime as dt

def read_meteoSatData(filename):

    df = pandas.read_csv(filename)    
    times=[]
    #temp_ave=[]
    pressure=[]

    for i in range(0,len(df['date'])):
        splitted_date=df['date'][i].split('-')
        times.append(dt.datetime(int(splitted_date[0]), int(splitted_date[1]),  int(splitted_date[2]) ) )      
        #temp_ave.append(df['tavg'][i])
        pressure.append(df['pres'][i]-29) # correzione a 242m slm (solo altitudine)

    return times,pressure

def read_xcfTempDatas(dir_path):  ### simo read XCF temp sensor

    fList = glob.glob(dir_path + "/temp*.txt")

    time=[]
    temp=[]
    hum=[]
    
    for filename in fList:
        f=open(filename)
    
        for line in f:
            
           time.append(dt.datetime.fromtimestamp(  (float(line[:-1].split()[0]))   ))
           temp.append(float(line[:-1].split()[1]))
           hum.append(float(line[:-1].split()[2])  )
           #print('time=',time," temp= ",temp,' humidity= ',hum)

    return time,temp,hum   







def air_density(temperature, pressure, humidity):
    psat = 6.1078*math.pow(10,7.5*temperature/(temperature+237.3))
    pv = humidity*psat
    pd = pressure-pv
    density = (pd*0.0289652 + pv*0.018016)/(8.31446*(273.15+temperature))
    return density


def dryAir_density(temperature, pressure):

    millibarToPascal=100
    P=pressure*millibarToPascal 
    M=0.0289652 #  molar mass of dry air, in kg⋅mol−1.
    R= 8.31446261815324# in J⋅K−1⋅mol−1
    T=temperature+273.1
    density = (P*M)/(R*T)
    #print("temp=",temperature," pressure=",pressure," dry Air density=",density)
    return density

def interpolate_mu(myE, plot=False):

    x_energy=[-0.1,1.,1.5,2.,3.,3.2,4.,5.,6.,8.,10,20,50]  # KeV
    mu_su_rho=[5e3,3.6e3,1.191e3,5.279e2,1.625e2,1.34e2,7.778e1,3.931e1,2.27e1,9.446,5.120, 7.779e-1,2.080E-01]


    f = interpolate.interp1d(x_energy, mu_su_rho, kind='cubic' )

    # get mu/rho @ myE
    #myE=2.7
    my_mu=f(myE)

    #plotting
    if plot==True:    
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




def attenuation_vs_d(E, d,  plot=False, temp=15,pressure=1013,hum=1.0 ):
 
    
    # air attenuation
    #air_rho=1.21e-3 # [gcm-3]
    air_rho=dryAir_density(temp, pressure)*1e-3 # from kg/m3 to g/cm3  
    #rint("air density std= ",air_rho)
    my_mu= interpolate_mu(E)     
    air_att=np.exp(-(d)*my_mu*air_rho)

    if plot==True:
        plt.figure()
        plt.semilogy(d,air_att, '-b',label='air attenuation at '+str(E)+' keV')
        
        plt.ylabel('attenuation')
        plt.xlabel('d [cm]')
        plt.legend()

        plt.show()
    
    return air_att




if __name__ == "__main__":


    Ebins=np.arange(2,10,0.1)
    E=5.9
    d=np.arange(1,20,0.1)
    att=attenuation_vs_d(E,d,plot=True)


    temps=np.arange(20,32,0.5)
    att=[]
    for t in temps:
       rel_att=attenuation_vs_d(E,2,temp=t)/attenuation_vs_d(E,2,temp=temps[0])
       att.append(rel_att)
    
    plt.figure()
    plt.plot(temps,att, '-b',label='relative attenuation vs T @2cm, 1013hPa, humidity=1')
    plt.show()
    


    #temperature = 15 # Celsius
    #pressure = 1013 # hPa
    #humidity = 1.0 # relative humidity [0-1]
    #density = air_density(temperature, pressure, humidity)
    #print(f"At {temperature}°C, {pressure}hPa, and {humidity:.0%} humidity, the air density is {density:.3f} kg/m^3")


    
