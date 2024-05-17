import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('--energy', type=float,  help='Energy [eV]', required=False, default=None)
parser.add_argument('--crystal', type=str,  help='crystal name: e.g. Si111', required=False, default=None)

args = parser.parse_args()
E = args.energy
crystal = args.crystal

InSb_111_E = [1740,2042,2166,2293,2622,2984,3692,4466,4511,4952,5415,5899,6930,7478,8048,8639,9442,9713,9886,9989]
InSb_111_k = [0.715,0.142,0.0457,0.0034,0.0702,0.238,0.472,0.573,0.582,0.658,0.721,0.771,0.843,0.869,0.890,0.906,0.924,0.928,0.931,0.933]

Ge_111_E = [2042,2166,2293,2622,2984,3692,4466,4511,4952,5415,5899,6930,7478,8048,8639,9442,9713,9886,9989]
Ge_111_k = [0.577,0.339,0.176,0.0036,0.06,0.338,0.557,0.567,0.647,0.710,0.760,0.831,0.856,0.877,0.894,0.913,0.918,0.921,0.922]

Si_111_E = [2042,2166,2293,2622,2984,3692,4466,4511,4952,5415,5899,6930,7478,8048,8639,9442,9713,9886,9989]
Si_111_k = [0.790,0.494,0.286,0.0280,0.0252,0.291,0.525,0.535,0.621,0.689,0.742,0.819,0.846,0.868,0.887,0.906,0.909,0.913,0.915]

Si_220_E = [3692,4466,4511,4952,5415,5899,6930,7478,8048,8639,9442,9713,9886,9989,14990,17480,21180,25270,29780]
Si_220_k = [0.420,0.0068,0.0023,0.0691,0.205,0.331,0.523,0.593,0.651,0.699,0.749,0.764,0.772,0.778,0.905,0.930,0.953,0.967,0.976]

Ge_422_E = [5415,5899,6273,6404,6930,7478,7656,8048,8398,8639,8912,9252,9442,9713,9886,9989,14990,17480,21180,25270,29780]
Ge_422_k = [0.955,0.598,0.288,0.325,0.124,0.0048,0.0017,0.0539,0.121,0.167,0.216,0.272,0.303,0.341,0.365,0.378,0.683,0.773,0.849,0.896,0.929]

hc = 1.240e-9 # keV m

dd_InSb111 = 7.481e-10
dd_Ge111 = 6.532e-10
dd_Si111 = 6.271e-10
dd_Si220 = 3.840e-10
dd_Ge422 = 2.31e-10

def fun(x,A,B):
    #x = np.radians(x)
    #B=1.5
    y = B*np.cos((2*np.arcsin(A*1000/x)))**2
    return y

def polarization(k):
    return (1-k)/(1+k)

def polarization_from_k(energy,par):
    k = fun(energy,A=par[0],B=par[1])
    return polarization(k)

def polarizaion_interpolation(energy,xarray,yarray):
    k = np.interp(x=energy,xp=xarray,fp=yarray)
    return polarization(k)

def kappa(P):
    return (1-P)/(1+P)

x_InSb111 = np.linspace(np.min(InSb_111_E),3800,1000)
x_Ge111 = np.linspace(np.min(Ge_111_E),np.max(Ge_111_E)/2,1000)
x_Si111 = np.linspace(np.min(Si_111_E),np.max(Si_111_E)/2,1000)
x_Si220 = np.linspace(np.min(Si_220_E),7000,1000)
x_Ge422 = np.linspace(6400,9300,1000)
# x_InSb111 = np.linspace(np.min(InSb_111_E),np.max(InSb_111_E),1000)
# x_Ge111 = np.linspace(np.min(Ge_111_E),np.max(Ge_111_E)/2,1000)
# x_Si111 = np.linspace(np.min(Si_111_E),np.max(Si_111_E)/2,1000)
# x_Si220 = np.linspace(np.min(Si_220_E),7000,1000)
# x_Ge422 = np.linspace(6400,9300,1000)

i=9
p_InSb111, c_InSb111 = curve_fit(fun, xdata = InSb_111_E[:i], ydata = InSb_111_k[:i],p0=[hc/dd_InSb111,1.5])
p_Ge111, c_Ge111 = curve_fit(fun, xdata = Ge_111_E[:i], ydata = Ge_111_k[:i],p0=[hc/dd_Ge111,1.5])
p_Si111, c_Si111 = curve_fit(fun, xdata = Si_111_E[:i], ydata = Si_111_k[:i],p0=[hc/dd_Si111,1.5])
p_Si220, c_Si220 = curve_fit(fun, xdata = Si_220_E[:i], ydata = Si_220_k[:i],p0=[hc/dd_Si220,1.5])
p_Ge422, c_Ge422 = curve_fit(fun, xdata = Ge_422_E[3:i], ydata = Ge_422_k[3:i],p0=[hc/dd_Ge422,1.5])

plt.figure()
plt.plot(InSb_111_E,InSb_111_k,marker='o')
plt.grid('both')
plt.title('InSb 111')
plt.xlabel('Energy eV')
plt.ylabel('k')
plt.fill_between(np.linspace(np.min(InSb_111_E),np.max(InSb_111_E),1000),0,kappa(0.98),alpha=0.5,color='green')
plt.axhline(kappa(0.98),label=' P 98%')
plt.axvline(x=2282)
plt.plot(x_InSb111,fun(x_InSb111,A=p_InSb111[0],B=p_InSb111[1]))
if crystal=='InSb111':
    plt.axvline(x=E,color='red')

plt.figure()
plt.plot(Ge_111_E,Ge_111_k,marker='o')
plt.grid('both')
plt.title('Ge 111')
plt.xlabel('Energy eV')
plt.ylabel('k')
plt.fill_between(np.linspace(np.min(Ge_111_E),np.max(Ge_111_E),1000),0,kappa(0.98),alpha=0.5,color='green')
plt.axhline(kappa(0.98),label=' P 98%')
plt.plot(x_Ge111,fun(x_Ge111,A=p_Ge111[0],B=p_Ge111[1]))
if crystal=='Ge111':
    plt.axvline(x=E,color='red')

plt.figure()
plt.plot(Si_111_E,Si_111_k,marker='o')
plt.grid('both')
plt.title('Si 111')
plt.xlabel('Energy eV')
plt.ylabel('k')
plt.fill_between(np.linspace(np.min(Si_111_E),np.max(Si_111_E),1000),0,kappa(0.98),alpha=0.5,color='green')
plt.axhline(kappa(0.98),label=' P 98%')
plt.plot(x_Si111,fun(x_Si111,A=p_Si111[0],B=p_Si111[1]))
if crystal=='Si111':
    plt.axvline(x=E,color='red')

plt.figure()
plt.plot(Si_220_E,Si_220_k,marker='o')
plt.grid('both')
plt.title('Si 220')
plt.xlabel('Energy eV')
plt.ylabel('k')
plt.fill_between(np.linspace(np.min(Si_220_E),np.max(Si_220_E),1000),0,kappa(0.98),alpha=0.5,color='green')
plt.axhline(kappa(0.98),label=' P 98%')
plt.plot(x_Si220,fun(x_Si220,A=p_Si220[0],B=p_Si220[1]))
if crystal=='Si220':
    plt.axvline(x=E,color='red')

plt.figure()
plt.plot(Ge_422_E,Ge_422_k,marker='o')
plt.grid('both')
plt.title('Ge 422')
plt.xlabel('Energy eV')
plt.ylabel('k')
plt.fill_between(np.linspace(np.min(Ge_422_E),np.max(Ge_422_E),1000),0,kappa(0.98),alpha=0.5,color='green')
plt.axhline(kappa(0.98),label=' P 98%')
plt.plot(x_Ge422,fun(x_Ge422,A=p_Ge422[0],B=p_Ge422[1]))
if crystal=='Ge422':
    plt.axvline(x=E,color='red')

if E is not None:
    if crystal == 'InSb111':
        par = p_InSb111
        e_array = InSb_111_E
        k_array = InSb_111_k
    if crystal == 'Ge111':
        par = p_Ge111
        e_array = Ge_111_E
        k_array = Ge_111_k
    if crystal == 'Si111':
        par = p_Si111
        e_array = Si_111_E
        k_array = Si_111_k
    if crystal == 'Si220':
        par = p_Si220
        e_array = Si_220_E
        k_array = Si_220_k
    if crystal == 'Ge422':
        par = p_Ge422
        e_array = Ge_422_E
        k_array = Ge_422_k
    
    POL = np.round(polarization_from_k(energy=E,par=par)*100,2)
    POL_i = np.round(polarizaion_interpolation(energy=E,xarray=e_array,yarray=k_array)*100,2)
    print(f'\n****************\nPolarization for E={E} eV and crystal {crystal} = {POL} % or {POL_i} % with interpolation\n****************\n')




plt.show()
