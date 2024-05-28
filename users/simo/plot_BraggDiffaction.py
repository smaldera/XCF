from matplotlib import pyplot as plt
import numpy as np


# approximate pol from thetaBragg
def computePol_thetaB(thetaB):
    # formula approssimata
    thetaB=np.deg2rad(thetaB)
    P=100.*(1-np.cos(2.*thetaB )**2  )/(  1+np.cos(2.*thetaB)**2 )
    return P



def thetaB_E(E,dd):
    h=2.*np.pi*6.582119563e-16
    c=299792458.0*1e10
    sinTheta=h*c/(dd*E)
    thetaRad=np.arcsin(sinTheta)
    return np.rad2deg(thetaRad)
   # return sinTheta

def EB_theta(thetaB,dd):
    h=2.*np.pi*6.582119563e-16
    c=299792458.0*1e10
    E=h*c/(dd*np.sin(np.deg2rad(thetaB)))
    return E


#E=2697
#pol_formula= computePol_thetaB(thetaBragg)

dd_Si220=3.840
dd_Ge111=6.532

dd=dd_Si220

E=np.linspace(1000,20000,1000)
#print(E)
plt.plot(E,thetaB_E(E,dd),'rp')

plt.figure(2)
plt.plot(E,computePol_thetaB(thetaB_E(E,dd)),'bp')

plt.figure(3)
theta=np.linspace(0,90,900)
plt.plot(theta,EB_theta(theta,6.532) ,'kp')
plt.grid()



plt.show()
