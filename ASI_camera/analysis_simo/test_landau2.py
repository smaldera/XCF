import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

#from landaupy import landau
import fit_histogram as fh
import utils_v2 as al




data=np.load('trackE.npz')
bins=data['bins']
counts=data['counts']


plt.hist(bins[:-1], bins =bins, weights=counts, histtype = 'step')
x=np.arange(1,26,0.1)
plt.plot(x, fh.myLandau(x, 9,    1.8,  600))

plt.show()

