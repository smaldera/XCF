import numpy as np
from matplotlib import pyplot as plt

plt.rc('axes', axisbelow=True)

def Gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def plotRep(x, lab,color='r'):
    fs = 14
    ns = 3.
    w, bins = np.histogram(x, 12, range=[x.min(), x.max()])
    print(bins)
    print(w)
    sigma = np.std(x, ddof=1)
    mean = np.mean(x)
    print("sigma=",sigma)
    print("mean=",mean)
     
    dots = np.linspace(mean-ns*sigma, mean+ns*sigma, 300)
    fig = plt.figure()
    plt.plot(dots, Gauss(dots, w.max(), mean, sigma))

    leg="mean= "+str(round(mean, 3))+" \n"+r"$\sigma$= "+str(round(sigma, 3))+" \n"+r"$\sigma$/mean= "+str(round(sigma/mean, 3))
    plt.grid()
    #plt.hist(bins[:-1], bins=bins, weights = w, histtype='step',label=leg  )
    plt.hist(bins[:-1], bins=bins, weights = w, alpha=0.9, label=leg, color=color, histtype='bar'  )
   
    plt.xlabel(lab, fontsize=fs)
    plt.ylabel('Counts', fontsize=fs)
    plt.xlim(mean-ns*sigma, mean+ns*sigma)
    #plt.grid()
    plt.legend()
    #plt.title(f'{lab} distribution - mean: {round(mean, 3)}, sigma: {round(sigma, 3)} sigma/mean: {round(sigma/mean,3)} ', fontsize=fs)
    plt.title(f'{lab} distribution ', fontsize=fs+2)
    plt.savefig(lab)
                                                   
  

leng = np.array([0.6720, 0.6751, 0.6731, 0.6713, 0.6742, 0.6725, 0.6688, 0.6726, 0.6703, 0.6727, 0.6701, 0.6695, 0.6714, 0.6701, 0.6682, 0.6714, 0.6734, 0.6695, 0.6745, 0.6711, 0.6684, 0.6699, 0.6691, 0.6728, 0.6708, 0.6712, 0.6685, 0.6658, 0.6706, 0.6706])
gain = np.array([16466.5, 16439.9, 16451.1, 16433.5, 16422.2, 16419.2, 16394.8, 16400.8, 16387.6, 16381.0, 16378.8, 16362.7, 16333.3, 16348.8, 16331.6, 16336.0, 16343.5, 16277.1, 16269.5, 16275.8, 16300.3, 16291.8, 16272.0, 16265.1, 16262.3, 16265.4, 16213.3, 16234.0, 16241.1, 16215.9])
rate = np.array([26.476, 26.406, 26.261, 27.020, 26.214, 26.324, 26.358, 27.025, 26.351, 26.138, 26.633, 26.235, 26.362, 26.007, 26.196, 26.673, 26.826, 26.435, 26.028, 26.325, 26.335, 26.547, 26.323, 26.309, 26.397, 26.392, 26.910, 26.160, 26.903, 26.249])

print(len(gain))
print(len(leng))
print(len(rate))

plotRep(leng, 'Track length',color='r')
plotRep(gain, 'Gain',color='b')
plotRep(rate, 'Rate',color='m')
plt.show()


