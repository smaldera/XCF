import numpy as np
import time



#this class is to debug the code without needing to have a camera connected if problems ara arising and are not related to the camera
class FakeCam:
    def __init__(self):
        self.NBINS = 16384  # n.canali ADC (2^14)
        self.XBINS = 2822
        self.YBINS = 4144


    def capture(self):
        data = np.zeros((self.XBINS,self.YBINS))
        value_to_set = 2000
        lines_to_set = 20

        data[100:100 + lines_to_set, :] = value_to_set
        #time.sleep(1)
        return data
