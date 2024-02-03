import numpy as np
from matplotlib import pyplot  as plt




filename='XCF/software/XCF/ASI_camera/GUI/temps.npz'
data=np.load(filename)

#print (data['temp'])

plt.plot(data['time']-data['time'][0],data['temp'],'ro')


plt.show()
