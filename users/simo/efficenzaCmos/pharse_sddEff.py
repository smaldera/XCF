import pandas
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np

df = pandas.read_csv('sdd_eff.txt',skipinitialspace=True)

sorted_df=df.sort_values(by='energy',ascending=True)


f = interpolate.interp1d(sorted_df['energy'],sorted_df['eff'] , kind='cubic' )



x=np.arange(1,30,0.1)
y=f(x)



plt.plot(sorted_df['energy'],sorted_df['eff'],'o-')
plt.plot(x,y,'*r')




plt.show()

