import numpy as np
import argparse
from astropy.io import fits
from XCFreader import GRun
from XCFGraph import GFigure, GPDMaps
from XCFgpd import GPD

parser = argparse.ArgumentParser()
parser.add_argument("--deg0", type=str, help = "Path to fits files with angle 0deg", required=True)
parser.add_argument("--deg90", type=str, help = "Path to fits files with angle 90deg", required=True)
parser.add_argument("--out", "--output-folder", type=str, help = "Path to output dir", default='./')
parser.add_argument("--s", "--name-suffix", type=str, help = "File name suffix (result SpMap_suffix.fits)", default=None)
parser.add_argument("--d", "--diagnostic", type=str, help = "Shows hepful data plots", default=False)
args = parser.parse_args()

class SpuriousMap():

    def __init__(self, path0, path90, outdir='./', suffix=None):
        self.Run_00 = GRun(path0)
        self.Run_90 = GRun(path90)
        if suffix is None:
            self.Path2File = outdir + 'SpMap.fits'
        else:
            self.Path2File = outdir + 'SpMap_' + suffix + '.fits'
        
        par_00, cov_00, gaussian_00, chi2_00 = self.Run_00.CutUnderPeak()
        par_90, cov_90, gaussian_90, chi2_90 = self.Run_90.CutUnderPeak()
        self.std_Cut00 = [par_00, cov_00, gaussian_00, chi2_00]
        self.std_Cut90 = [par_90, cov_90, gaussian_90, chi2_90]
        self.q_00 = self.Run_00.GetQ()
        self.u_00 = self.Run_00.GetU()
        self.q_90 = self.Run_90.GetQ()
        self.u_90 = self.Run_90.GetU()

    def DoQU(self):
        self.q_00 = self.Run_00.GetQ()
        self.u_00 = self.Run_00.GetU()
        self.q_90 = self.Run_90.GetQ()
        self.u_90 = self.Run_90.GetU()

    def DiagnosticGraph(self):
        pha0 = self.Run_00.GetPha()
        pha90 = self.Run_90.GetPha()
        DCanvas = GFigure(1,2)
        DCanvas.AddHist(pha0, 100, [pha0.min(),pha0.max()], label=r'$Pha_{0}$ distribution', position=0)
        DCanvas.AddHist(pha90, 100, [pha90.min(),pha90.max()], label=r'$Pha_{90}$ distribution', position=1)
        DCanvas.AxesLabels(xlabel=r'Pha', ylabel='counts', position=0)
        DCanvas.AxesLabels(xlabel=r'Pha', ylabel='counts', position=1)
        DCanvas.TurnOnLegend(position=0)
        DCanvas.TurnOnLegend(position=1)
        DCanvas.AxesLimits([0., 2*np.mean(pha0)], [None, None], 0)
        DCanvas.AxesLimits([0.,2*np.mean(pha90)], [None, None], 1)

    def HotSpot(self, xc, yc, a, b):
        self.Run_00.EllipticCut(xc, yc, a, b)
        self.Run_90.EllipticCut(xc, yc, a, b)
        self.DoQU()

    def MeanAndSigma(self, q0, q90, sigma0, sigma90):
        mean_sp = ( np.mean(q0)+np.mean(q90) )/2.
        mean_int = ( np.mean(q0)-np.mean(q90) )/2.
        sigma = ( np.sqrt( sigma0**2 + sigma90**2 ) )/2.
        return mean_sp, mean_int, sigma
        
    def MeanSpuriousValue(self):
        #self.Run_00.EllipticCut(0.,0.,2.,2.,False)
        #self.Run_90.EllipticCut(0.,0.,2.,2.,False)
        physicalEvents_00 = self.Run_00.GetPhysicsMask()
        physicalEvents_90 = self.Run_90.GetPhysicsMask()
        sigmaQ_00, sigmaU_00 = self.Run_00.CalculateQUError( np.mean(self.q_00[physicalEvents_00]), np.mean(self.u_00[physicalEvents_00]), len(physicalEvents_00) )
        sigmaQ_90, sigmaU_90 = self.Run_90.CalculateQUError( np.mean(self.q_90[physicalEvents_90]), np.mean(self.u_90[physicalEvents_90]), len(physicalEvents_90) )
        Q, Qint, sQ = self.MeanAndSigma(self.q_00[physicalEvents_00], self.q_90[physicalEvents_90], sigmaQ_00, sigmaQ_90)
        U, Uint, sU = self.MeanAndSigma(self.u_00[physicalEvents_00], self.u_90[physicalEvents_90], sigmaU_00, sigmaU_90)

        MeanSpurious = np.sqrt( Q**2 +  U**2 )
        MeanIntrinsec = np.sqrt( Qint**2 +  Uint**2 )
        SigmaSpurious = ( np.sqrt( (Q*sQ)**2 + (U*sU)**2 ) )/MeanSpurious
        SigmaIntrinsec = ( np.sqrt( (Qint*sQ)**2 + (Uint*sU)**2 ) )/MeanIntrinsec
        print('-'*70+f'\nMEAN SPURIOUS MODULATION VALUE: {MeanSpurious:.4f} ± {SigmaSpurious:.4f}')
        print(f'MEAN Q: {Q:.4f} ± {sQ:.4f}')
        print(f'MEAN U: {U:.4f} ± {sU:.4f}')
        print(f'MEAN INTRINSEC MODULATION VALUE: {MeanIntrinsec:.4f} ± {SigmaIntrinsec:.4f}\n'+'-'*70+'\n')
        return MeanSpurious, SigmaSpurious, MeanIntrinsec, SigmaIntrinsec

    def MakeSpuriousMap(self, show=False, save=True):
        print("Making spurious map.\nBinned:300x300")
        Q_00 = self.Run_00.GetQMap()
        U_00 = self.Run_00.GetUMap()
        Q_90 = self.Run_90.GetQMap()
        U_90 = self.Run_90.GetUMap()
        C_00 = self.Run_00.GetCounts()
        C_90 = self.Run_90.GetCounts()

        Q_sp = (Q_00+Q_90)/2.
        U_sp = (U_00+U_90)/2.
        Q_sp_err = np.where( (C_00>1) & (C_90>1), 0.5*np.sqrt((2-Q_00**2)/(C_00-1)+(2-Q_90**2)/(C_90-1)), 0 )
        U_sp_err = np.where( (C_00>1) & (C_90>1), 0.5*np.sqrt((2-U_00**2)/(C_00-1)+(2-U_90**2)/(C_90-1)), 0 )
        SpuriousCanvas = GFigure(1, 2)
        
        mapQ = SpuriousCanvas.AddMap(Q_sp, vmin=-0.3, vmax=0.3, lim=7.5, position=0)
        SpuriousCanvas.AddColorBar(mapQ, 'Q', position=0)
        SpuriousCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        SpuriousCanvas.SetTitle(r'$Q_{sp}$', position=0)
        
        mapU = SpuriousCanvas.AddMap(U_sp, vmin=-0.3, vmax=0.3, lim=7.5, position=1)
        SpuriousCanvas.AddColorBar(mapU, 'U', position=1)
        SpuriousCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=1)
        SpuriousCanvas.SetTitle(r'$U_{sp}$', position=1)
        
        ErrorCanvas = GFigure(1,3)
        
        mapC = ErrorCanvas.AddMap(C_00+C_90, vmin=0, vmax=C_00.max()*2, lim=7.5, position=0)
        ErrorCanvas.AddColorBar(mapC, 'Counts', position=0)
        ErrorCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        ErrorCanvas.SetTitle(r'$Counts_{0}\,+\,Counts_{90}$', position=0)

        mapQ_err = ErrorCanvas.AddMap(Q_sp_err, vmin=-0.3, vmax=0.3, lim=7.5, position=1)
        ErrorCanvas.AddColorBar(mapQ_err, r'$\sigma_Q$', position=1)
        ErrorCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=1)
        ErrorCanvas.SetTitle(r'$Q_{Spurious}\, error map$', position=1)

        mapU_err = ErrorCanvas.AddMap(U_sp_err, vmin=-0.3, vmax=0.3, lim=7.5, position=2)
        ErrorCanvas.AddColorBar(mapU_err, r'$\sigma_U$', position=2)
        ErrorCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=2)
        ErrorCanvas.SetTitle(r'$U_{Spurious}\, error map$', position=2)

        M = np.sqrt(Q_sp**2 + U_sp**2)
        psi = 0.5*np.rad2deg(np.arctan2(U_sp, Q_sp))

        QCanvas = GFigure(1,2)
        Q_sp.reshape(np.shape(Q_sp)[0]*np.shape(Q_sp)[1])
        istQ = QCanvas.AddHist(Q_sp, 100, [-2.,2.], label=r'$Q_{sp}$ distribution', position=0)
        QCanvas.AxesLabels(xlabel=r'Q_{sp}', ylabel='counts', position=0)
        QCanvas.SetTitle(r'$Q_{sp}$', position=0)

        Q_sp_err.reshape(np.shape(Q_sp_err)[0]*np.shape(Q_sp_err)[1])
        istQerr = QCanvas.AddHist(Q_sp_err/Q_sp, 100, [0.,10], label=r'rel $ Qerr_{sp}$', position=1)
        QCanvas.AxesLabels(xlabel=r'$|\sigma_Q/Q|$', ylabel='counts', position=1)
        QCanvas.SetTitle(r'Relative $Qerr_{sp}$', position=1)

        UCanvas = GFigure(1,2)
        U_sp.reshape(np.shape(U_sp)[0]*np.shape(U_sp)[1])
        istU = UCanvas.AddHist(U_sp, 100, [-2.,2.], label=r'rel $U_{sp}$ distribution', position=0)
        UCanvas.AxesLabels(xlabel=r'U_{sp}', ylabel='counts', position=0)
        UCanvas.SetTitle(r'$U_{sp}$', position=0)

        U_sp_err.reshape(np.shape(U_sp_err)[0]*np.shape(U_sp_err)[1])
        istUerr = UCanvas.AddHist(np.abs(U_sp_err/U_sp), 100, [0.,10], label=r'$Uerr_{sp}$', position=1)
        UCanvas.AxesLabels(xlabel=r'$|\sigma_U/U|$', ylabel='counts', position=1)
        UCanvas.SetTitle(r'Relative $Uerr_{sp}$', position=1)

        ModulationCanvas = GFigure(1, 2)
        mapM = ModulationCanvas.AddMap(M, vmin=0., vmax=1, lim=7.5, position=0)
        ModulationCanvas.AddColorBar(mapM, 'M', position=0)
        ModulationCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        ModulationCanvas.SetTitle(r'$M_{sp}$', position=0)

        M_cutted = ModulationCanvas.EdgeCut(M, 7)
        M = M.reshape(np.shape(M)[0]*np.shape(M)[1])
        ModulationCanvas.AddHist(M, 150, [0., 2.], label='Spurious modulation', position=1)
        ModulationCanvas.AddHist(M_cutted, 150, [0., 2.], label='Spurious modulation', position=1)
        ModulationCanvas.AxesLabels(xlabel=r'$M_{sp}$', ylabel='Frequency', position=1)
        ModulationCanvas.SetTitle(r'$M_{sp} distribution$', position=1)
        ModulationCanvas.TurnOnLegend(position=1)
        ModulationCanvas.DrawVline(self.meanMDP, 0, 4000, 'mean MDP', position=1)
        ModulationCanvas.AxesLimits([0,2.], [None, None], 1)

        AngleCanvas = GFigure(1,2)
        mapPsi = AngleCanvas.AddMap(psi, vmin=-90., vmax=90., lim=7.5, position=0)
        AngleCanvas.AddColorBar(mapPsi, r'$\psi [deg]$', position=0)
        AngleCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        AngleCanvas.SetTitle(r'$\psi_{sp}$', position=0)

        psi_cutted = AngleCanvas.EdgeCut(psi, 7)
        psi = psi.reshape(np.shape(psi)[0]*np.shape(psi)[1])
        AngleCanvas.AddHist(psi, 180, [-90, 90], label='Spurious Angle', position=1)
        AngleCanvas.AddHist(psi_cutted, 180, [-90, 90], label='Spurious Angle Cutted', position=1)
        AngleCanvas.AxesLabels(xlabel=r'$\psi_{sp}$', ylabel='Frequency', position=1)
        AngleCanvas.SetTitle(r'$\psi_{sp}$ distribution', position=1)
        AngleCanvas.TurnOnLegend(position=1)

        print("Map done")
        if show:
            SpuriousCanvas.GDraw()
        if save:
            self.SaveMaps(Q_sp, U_sp)
        return Q_sp, U_sp
    
    def MdpMap(self, nBins=100):
        print('\n----------------------MDP map added----------------------')
        MDP = GFigure(1,2)
        MDP.SetFigureTitle(suptitle='MDP Study')
        m = 4.29/np.sqrt(self.Run_00.GetCounts()+self.Run_90.GetCounts())
        self.meanMDP = np.mean(m[np.where(np.isinf(m)==False)])
        mapId = MDP.AddMap(m, vmax = m.max(), vmin = 0., lim=7.5, position=0)
        MDP.AddColorBar(mapId, 'M', 0)
        MDP.AxesLabels('AbsX [mm]', 'AbsY [mm]', position=0)
        MDP.SetTitle('MDP Map', position=0)
        m = m.reshape(np.shape(m)[0]*np.shape(m)[1])
        m[np.where(np.isinf(m))] = -1
        MDP.AddHist(m, nBins, [0.,1.], label='Modulation', position=1)
        MDP.AxesLabels(r'$\mu \cdot MDP$', 'Entries', position=1)
        MDP.SetTitle('M distribution', position=1)
        MDP.TurnOnGrid(1)
        MDP.GDraw()
        print(f'Mean MDP value {self.meanMDP}')
        print('----------------------Done with MDP----------------------\n')

    def SaveMaps(self, Q_sp, U_sp, numGPD=35):
        useful = [self.std_Cut00[0][1]-self.std_Cut00[0][2], self.std_Cut00[0][1]+self.std_Cut00[0][2]]
        hdr = fits.Header()
        hdr['GPD_NUM'] = numGPD
        hdr['PHA_BIN'] = f"PHI_MIN, PHI_MAX"
        hdr['Comment'] = 'Q and U spurious map'

        hdu0 = fits.PrimaryHDU(data=useful, header=hdr)
        hdu1 = fits.ImageHDU(data=Q_sp, name='Q_SP')
        hdu2 = fits.ImageHDU(data=U_sp, name='U_SP')
        hdu3 = fits.ImageHDU(data=np.sqrt(Q_sp**2 + U_sp**2), name='M_SP')
        hdu4 = fits.ImageHDU(data=(self.Run_00.GetCounts()+self.Run_90.GetCounts()), name='COUNTS')

        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        hdul.writeto(self.Path2File, overwrite=True)
        print('\nFILE SAVED')
        self.PrintFileInfo()

    def PrintFileInfo(self):
        hdul = fits.open(self.Path2File)
        print(hdul.info())
        
    def ShowBoth(self):
        Canvas00 = GPDMaps(self.Run_00)
        Canvas90 = GPDMaps(self.Run_90)
        Canvas00(False)
        Canvas90(False)
        Canvas00.AddFunctionPlot([self.std_Cut00[0][1]-4*self.std_Cut00[0][2], self.std_Cut00[0][1]+4*self.std_Cut00[0][2]], self.std_Cut00[2], self.std_Cut00[0], sigma=np.sqrt(np.diag(self.std_Cut00[1])), label = 'Peak Cut', chi2 = self.std_Cut00[3], position=1)
        Canvas90.AddFunctionPlot([self.std_Cut90[0][1]-4*self.std_Cut90[0][2], self.std_Cut90[0][1]+4*self.std_Cut90[0][2]], self.std_Cut90[2], self.std_Cut00[0], sigma=np.sqrt(np.diag(self.std_Cut90[1])), label = 'Peak Cut', chi2 = self.std_Cut90[3], position=1)
        Canvas00.GDraw()
        Canvas90.GDraw()

    def TargetMdp(self, mdp):
        N = (4.29/mdp)**2
        BinContent = np.mean(self.Run_00.GetCounts()+self.Run_90.GetCounts())
        M = np.shape(self.Run_00.GetCounts())[0]
        target = int(N/BinContent)
        print(f"\n\nYou should aggregate {target} bins. So {M/target}x{M/target} bins.")
        print("------------------------------REBINNING ALL MAPS------------------------------\n")
        self.Run_00.RebinAll(int(M/target))
        self.Run_90.RebinAll(int(M/target))


if __name__ == '__main__':
    Sp = SpuriousMap(args.deg0, args.deg90, args.out, args.s)
    Sp.TargetMdp(0.08)
    Sp.MdpMap()
    spurious, sigma, _, _ = Sp.MeanSpuriousValue()
    if args.d:
        Sp.DiagnosticGraph()
    Sp.MakeSpuriousMap(True)
