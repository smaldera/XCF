import numpy as np
from abc import ABC, abstractmethod
from cycler import cycler
from matplotlib import pyplot as plt
import XCFgpd as gpd

class GCanvas(ABC):

    def __init__(self, fontsize=16):
        plt.ion()
        self.EnvSettings()
        self.DotSize = 12
        self.LegendSize = fontsize+2
        self.LineWidth = 2
        self.ThicksSize = fontsize
        self.FontSize = fontsize
        self.TitleSize = fontsize+2
        self.cmap = 'viridis'
        self.contour_cmap = 'YlOrRd'
        self.palette = ['#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
        self.ResetAround()

    @abstractmethod
    def AddColorBar(self):
        # Adds an Colorbar
        pass
    
    @abstractmethod
    def AddContour(self):
        # Adds a Contour Plot
        pass

    @abstractmethod
    def AddErrorbar(self):
        # Adds one scatter plot
        pass

    @abstractmethod
    def AddExternalHist(self):
        # Adds one scatter plot
        pass

    @abstractmethod
    def AddFunctionPlot(self):
        # Adds F plot
        pass

    @abstractmethod
    def AddHist(self):
        # Adds a hist
        pass

    @abstractmethod
    def AddMap(self):
        # Adds a Img
        pass

    @abstractmethod
    def AddPlot(self):
        # Adds a Plot
        pass

    @abstractmethod
    def AddScatter(self):
        # Adds a Scatter
        pass

    @abstractmethod
    def AxesLabels(self):
        # Renames the labels on axes x, y
        pass

    @abstractmethod
    def AxesLimits(self):
        # Sets Axes Limits
        pass

    @abstractmethod
    def AxesTicks(self):
        # Sets Axes Limits
        pass

    @abstractmethod
    def GClose(self):
        # Closes target
        pass

    @abstractmethod
    def SetFigureTitle(self):
        # Sets Suptitle
        pass

    @abstractmethod
    def SetTitle(self):
        # Sets Axes Title
        pass

    @abstractmethod
    def TurnOnGrid(self):
        # Sets Grid
        pass

    @abstractmethod
    def TurnOnLegend(self, position):
        # Sets Legend
        pass

    def EnvSettings(self):
        myGPD = gpd.GPD()
        x_, y_, _, _ = myGPD.MapLimits()
        self.xShape = x_
        self.yShape = y_
        return x_, y_

    def GDraw(self):
        plt.show(block=True)

    def ResetAround(self):
        plt.rc('axes', prop_cycle=cycler('color', self.palette), linewidth=1)
        plt.rc('xtick', labelsize=self.ThicksSize)
        plt.rc('ytick', labelsize=self.ThicksSize)

    def SetColormap(self, cm):
        self.cmap = cm

    def SetContourColors(self, cc):
        self.contour_cmap = cc

    def SetLegendSize(self, ls):
        self.LegendSize = ls
    
    def SetTickSize(self, ts):
        self.ThicksSize = ts
        self.ResetAround()
    
    def SetFontSize(self, fs):
        self.FontSize = fs
    
    def SetPalette(self, palette):
        self.palette = palette
        self.ResetAround()

    def SetTitleSize(self, ts):
        self.TitleSize = ts










# Subclass: Single Figure
class GFigure(GCanvas):

    def __init__(self, nRows=1, nColumns=1, shareX=False, shareY=False, figsize=(8,8)):
        super().__init__()
        self.figure, ax = plt.subplots(nrows=nRows, ncols=nColumns, sharex=shareX, sharey=shareY, figsize=figsize)
        if nRows*nColumns != 1:
            self.axes = ax.reshape(nRows*nColumns)
        else:
            self.axes = [ax, 1]

    def AddColorBar(self, mapId, label, position=0):
        cbar = self.figure.colorbar(mapId, ax=self.axes[position], shrink=0.42)
        cbar.set_label(label=label, fontsize=self.FontSize, weight='bold')

    def AddContour(self, x, y, z, levels=3, alpha=1, position=0):
        self.axes[position].contour(x, y, z, levels=levels, alpha=alpha, linewidths=self.LineWidth, cmap=self.contour_cmap)

    def AddErrorbar(self, x, y, sigmaX=0., sigmaY=0., fmt='.', label='ScatterPlot', position=0):
        sp = self.axes[position].errorbar(x, y, xerr=sigmaX, yerr=sigmaY, markersize=self.DotSize, markeredgecolor='black', linewidth=self.LineWidth, label=label, fmt=fmt)
        return sp

    def AddExternalHist(self, binsedges, weights, alpha=0.6, label='ExtHist', position=0):
        if len(weights)>100:
            edgecolor='None'
        else:
            edgecolor='black'
        hist = self.axes[position].hist(binsedges[:-1], bins=binsedges, weights=weights, label=label, alpha=alpha, edgecolor=edgecolor, linewidth=1)
        return hist

    def AddFunctionPlot(self, xLimits, function, parameters, label='Function', sigma=None, chi2=0, position=0):
        X = np.linspace(xLimits[0], xLimits[1], 200)
        Y = function(X, *parameters)
        plot = self.AddPlot(X, Y, '-', label=label, position=position)
        info = f"$\chi^2$ = {chi2:.1f}\n"
        for i in range(len(parameters)):
            if sigma is None:
                info += f"Par{i}: {parameters[i]:.2f}\n"
            else:
                info += f"Par{i}: {parameters[i]:.2f} ± {sigma[i]:.2f}\n"
        self.axes[position].text(1.1*xLimits[1], 0.7*Y.max(), info, fontsize=self.LegendSize-4, bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle="round,pad=0.3"))
        return plot

    def AddHist(self, x, nBins, limits, alpha=0.8, label='Histogram', normalized=False, position=0):
        h, b = np.histogram(x, bins=nBins, range=limits, density=normalized)
        hist = self.AddExternalHist(b, h, alpha=alpha, label=label, position=position)
        return hist
    
    def AddMap(self, xy, vmax=0.3, vmin=-0.3, lim=7.5, position=0):
        maps = self.axes[position].imshow(xy.T,  vmax=vmax, vmin=vmin, extent=[-lim, lim, -lim, lim], interpolation='nearest', cmap=self.cmap)
        return maps
    
    def AddPlot(self, x, y, fmt='.', label='Plot', position=0):
        pl = self.axes[position].plot(x, y, fmt, markersize=self.DotSize, markeredgecolor='black', linewidth=self.LineWidth, label=label)
        return pl
    
    def AddScatter(self, x, y, z=None, marker='.', s=12, label='Scatter', edgecolor='black', position=0):
        sc = self.axes[position].scatter(x,y,c=z,cmap=self.cmap, marker=marker, edgecolor=edgecolor, s=s, label=label)
        return sc
    
    def ApplyTightLayout(self, pad=1.08, h_pad=None, w_pad=None, rect=None):
        self.figure.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
    
    def AxesLabels(self, xlabel, ylabel, weight='bold', position=0):
        self.axes[position].set_xlabel(xlabel, fontsize=self.FontSize, fontweight=weight)
        self.axes[position].set_ylabel(ylabel, fontsize=self.FontSize, fontweight=weight)
    
    def AxesLimits(self, xlim=[None, None], ylim=[None, None], position=0):
        self.axes[position].set_xlim(left=xlim[0], right=xlim[1])
        self.axes[position].set_ylim(bottom=ylim[0], top=ylim[1])

    def AxesTicks(self, xticks=None, yticks=None, xlabels=None, ylabels=None, position=0):
        if xticks is None:
            if yticks is not None:
                self.axes[position].set_yticks(yticks, ylabels)
        elif yticks is None:
            self.axes[position].set_xticks(xticks, xlabels)
        else:
            self.axes[position].set_xticks(xticks, xlabels)
            self.axes[position].set_yticks(yticks, ylabels)

    def DrawEllipse(self, xc, yc, a, b, angle=0, label='', position=0):
        angle = np.radians(angle)
        theta = np.linspace(0, 2 * np.pi, 360)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        x_rot = xc + x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle) + yc
        self.AddPlot(x_rot, y_rot, '-', label=label, position=position)
    
    def DrawVline(self, x0, y0, y1, label='', position=0):
        self.axes[position].vlines(x0, y0, y1, colors='m', label=label)

    def EdgeCut(self, Z, edge=7):
        lim = 7.5
        dx = (2 * lim) / self.xShape
        dy = (2 * lim) / self.yShape
        x_start_idx = int((lim - edge) / dx)
        x_end_idx = int((lim + edge) / dx)
        y_start_idx = int((lim - edge) / dy)
        y_end_idx = int((lim + edge) / dy)
        Z_central = Z[x_start_idx:x_end_idx, y_start_idx:y_end_idx]
        return Z_central.flatten()

    def GClose(self):
        plt.close(self.figure)

    def SetEqual(self, position=0):
        self.axes[position].set_aspect('equal', adjustable='datalim')

    def SetFigureTitle(self, suptitle):
        self.figure.suptitle(suptitle, fontsize = self.TitleSize, weight = 'bold')

    def SetTitle(self, title, weight='bold', position=0):
        self.axes[position].set_title(title, fontsize=self.TitleSize, fontweight=weight)

    def TurnOnGrid(self, position=0):
        self.axes[position].grid()

    def TurnOnLegend(self, position=0):
        self.axes[position].legend(fontsize=self.LegendSize)










# sublclass that generates projections and pha maps
class GPDMaps(GFigure):

    def __init__(self, data, figsize=(8,8)):
        super().__init__(1, 3, figsize=figsize)
        self.data = data

    def __call__(self, projections=True, suptitle='title'):
        self.SetFigureTitle(suptitle=suptitle)
        self.ApplyTightLayout(pad=5.0, w_pad=3.0)
        self.CountMap()
        self.PhaHist()
        self.PhiHist()
        if projections:
            self.MakeProjections(suptitle=suptitle)

    def ContourCounts(self, position=0):
        c = self.data.GetCounts()
        x = np.linspace(-7.5, 7.5, self.xShape)
        y = np.linspace(7.5, -7.5, self.yShape)
        X, Y = np.meshgrid(x, y)
        self.AddContour(X, Y, c.T, position=position)

    def CountMap(self):
        c = self.data.GetCounts()
        mapId = self.AddMap(c.reshape(self.xShape, self.yShape), vmax = c.max(), vmin = 0., lim=7.5, position=0)
        self.AddColorBar(mapId, 'Counts', 0)
        self.AxesLabels('AbsX [mm]', 'AbsY [mm]', position=0)
        self.SetTitle('Counts Map', position=0)

    def GetProjection(self):
        return self.ProjCanvas

    def MakeProjections(self, nBins=300, limits=[-7.5, 7.5], suptitle=''):
        self.ProjCanvas = GFigure(1,2)
        self.ProjCanvas.SetFigureTitle(suptitle=suptitle)
        self.ProjCanvas.AddHist(self.data.GetAbsX(), nBins, limits, label='AbsX', position=0)
        self.ProjCanvas.AxesLabels('X [mm]', 'Entries', position=0)
        self.ProjCanvas.SetTitle('X projections', position=0)
        self.ProjCanvas.TurnOnGrid(0)
        self.ProjCanvas.AddHist(self.data.GetAbsY(), nBins, limits, label='AbsY', position=1)
        self.ProjCanvas.AxesLabels('Y [mm]', 'Entries', position=1)
        self.ProjCanvas.SetTitle('Y projections', position=1)
        self.ProjCanvas.TurnOnGrid(1)

    def PhaHist(self, nBins=200, limits=[0, 30000]):
        pha = self.data.GetPha()
        self.AddHist(pha, nBins, limits, label='Pha', normalized=True, position=1)
        self.AxesLabels(xlabel='Pha', ylabel='Normalized Counts', position=1)
        self.SetTitle('Pha Distribution', position=1)
        self.AxesLimits([0, None], position=1)
        self.TurnOnGrid(1)
        self.TurnOnLegend(1)

    def PhiHist(self, nBins=200, limits=[-180., 180.]):
        phi = self.data.GetPhi()
        self.AddHist(np.degrees(phi), nBins, limits, label='Phi2', position = 2)
        self.AxesLabels(xlabel='Phi2 [deg]', ylabel='Entries', position=2)
        self.SetTitle(f'Modulation: {(self.data.GetModulation()*100):.3}% , Psi: {np.degrees(self.data.GetPsi()):.3} deg', position=2)
        self.AxesLimits([-180, 180], position=2)
        self.TurnOnGrid(2)
        self.TurnOnLegend(2)

    def UpdateData(self, data):
        self.data = data
        plt.close(self.figure)
        self.__call__()








if __name__ == '__main__':
    myCanvas = GFigure(1, 1)
    myCanvas.SetTitle('Test 0: XCF graph')
    myCanvas.AddPlot([0,2,4], [1,2,3], '.-',label='Plot0')
    myCanvas.AddPlot([1,3,3], [1,2,3], '.-', label='Plot0')
    myCanvas.AddPlot([2,1,0], [4,1,2], '.-', label='Plot0')
    myCanvas.AddErrorbar([0,1,2], [0,1,2], 0.1, 0.2, label='Scatter0')
    myCanvas.AxesLabels('X [cm]', 'Y [cm]')
    myCanvas.TurnOnGrid()
    myCanvas.TurnOnLegend()

    myCanvas1 = GFigure(1, 1)
    myCanvas1.AddHist([1,2,1,1,1,2,2,3,1,2,0,4,4,3,2,6,7,8,9,0,7,8,9,4,5,1,6,9], 10, [0,10], label='Hist0')
    myCanvas1.AddHist([1,2,2,1,3,2,2,3,1,2,0,5,4,3,2,9,7,8,9,0,10,8,9,4,8,1,6,9], 10, [0,10], label='Hist0')
    myCanvas1.AddHist([1,4,2,4,3,4,4,4,1,2,1,5,4,3,2,5,7,8,9,0,10,5,9,5,8,5,6,5], 10, [0,10], label='Hist0')
    myCanvas1.SetTitle('Test 1: XCF hist')
    myCanvas1.AxesLabels('X [cm]', 'Y [cm]')
    myCanvas1.AxesTicks([3,6,9], None, ['A','B','C'])
    myCanvas1.AxesLimits([0,10])
    myCanvas1.TurnOnLegend()
    myCanvas1.GDraw()
