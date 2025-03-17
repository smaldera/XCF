import numpy as np
import astropy.io.fits as pf
from itertools import product
import XCFGraph as GGraph

class GPD():

    def __init__(self, GPD_number = 35, mapbins=300):
        self.GPD_number = GPD_number
        self.nRows = mapbins
        self.nColumns = mapbins
        self.OrzPitch = 0.050    # mm
        self.VertPitch = 0.0433  # mm
        self.AxLimit = 7.5       # mm
        x_, y_, _, _ = self.MapLimits()
        q = np.zeros(x_*y_)
        u = np.zeros(x_*y_)
        self.qSpurious = q.reshape(x_, y_)
        self.uSpurious = u.reshape(x_, y_)

    def GetRows(self):
        return self.nRows

    def GetColumns(self):
        return self.nColumns

    def GetGPDNumber(self):
        return self.GPD_number

    def GetLimits(self):
        return self.AxLimit

    def GetOrzPitch(self):
        return self.OrzPitch

    def GetQ(self):
        return self.qSpurious

    def GetU(self):
        return self.uSpurious
    
    def GetVertPitch(self):
        return self.VertPitch

    def SetSpuriousMap(self, qMap, uMap):
        self.qSpurious = qMap
        self.uSpurious = uMap

    def MapLimits(self):
        x_range = np.linspace(-self.AxLimit, self.AxLimit, self.nColumns+1)
        y_range = np.linspace(self.AxLimit, -self.AxLimit, self.nRows+1)
        x_ = len(x_range)-1
        y_ = len(y_range)-1
        return x_, y_, x_range, y_range

    #This function must be updated: it should have a repo of all SpMaps per Energy bin for each GPD
    def ReadSpurious(self, filePath, energy=2.28):
        data_map = pf.open(filePath)
        q = data_map[1].data
        u = data_map[2].data
        self.SetSpuriousMap(q, u)






class Track(GPD):

    def __init__(self, file_path, GPD_number=35):
        super().__init__(GPD_number)
        data = pf.open(file_path)
        self.events = data['EVENTS'].data
        
    def _ij2xy(self, ij):
        "Function to go from i (col) and j (row) to gpd absolute x and y in mm."
        if type(ij) is np.ndarray:
            i = np.array(ij[:, 1])
            j = np.array(ij[:, 0])
        else:
            i, j = ij[1], ij[0]
        y = self.AxLimit - j*self.VertPitch
        x = np.where(j%2==0, (i+0.5)*self.OrzPitch, i*self.OrzPitch ) - self.AxLimit
        if type(ij) is np.ndarray:
            return np.array(list(zip(y,x)))
        else:
            return (y, x)
        
    def ViewTrack(self, n):
        ev = self.events[n]
        (min_col, max_col, min_row, max_row) = (ev['MIN_CHIPX'],ev['MAX_CHIPX'],ev['MIN_CHIPY'],ev['MAX_CHIPY'])
        Nrows = max_row-min_row
        Ncols = max_col-min_col

        j_ = np.arange(min_row, max_row+1, 1)
        i_ = np.arange(min_col, max_col+1, 1)
        ij = np.array(list(product(j_, i_)))

        xy = self._ij2xy(ij)
        y = xy[:,0]
        x = xy[:,1]

        # Only main track will be considered
        ch=ev['PIX_PHAS_EQ']
        myCanvas = GGraph.GFigure(1,1, figsize=(Ncols*0.92, Nrows*0.81))
        myCanvas.SetTitle(f"Evnt number {n}. Baricenter: {ev['ABSX']:.3} mm, {ev['ABSY']:.3} mm")
        myCanvas.AddScatter(x, y, ch, 'h', 800, 'Event track')
        myCanvas.AddScatter(ev['BARX'],ev['BARY'], marker='*', s=600, label='Charge Baricenter')
        myCanvas.SetEqual()
        myCanvas.AxesLabels('X [mm]', 'Y [mm]')
        myCanvas.TurnOnLegend()
        myCanvas.GDraw()





    
if __name__ == '__main__':
    path = '/data2/XCF/gpd_data/GPD35/911_0000883/911_0000883_data_tmk5.fits'
    myTrack = Track(path)
    myTrack.ViewTrack(49)
