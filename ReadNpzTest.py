import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/XCF/Desktop/SaveNpz/11_06_24/Air_10kV_0.006mA_2024_6_11_10_24.npz"

data = np.load(path)
y = data['spectrum']
var = data['utils']

"""
    utils contains the useful parameters of the aquisition:
    utilData = [ livetime [s], FastCount [#], SlowCount [#], DeadTime [%], Rate [Hz], start unixtime [s], stop unixtime [s] ]
"""

x = np.linspace(0,8192,8193)
fig = plt.figure()
plt.plot(x[:-1],y)
plt.show()

print(var)
print(y)
print(len(y))
