from matplotlib import pyplot as plt
import mpld3
import numpy as np

x=np.arange(0,100)
y=x**2+4*x-3


fig = plt.figure(figsize = (18,8))
plt.plot(x,y,'or-')
#plt.show()
html_str = mpld3.fig_to_html(fig)
Html_file= open("index.html","w")
Html_file.write(html_str)
Html_file.close()


