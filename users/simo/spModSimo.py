#***********************************************************************
# Copyright (C) 2017 the Imaging X-ray Polarimetry Explorer (IXPE) team.
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#***********************************************************************
from __future__ import print_function, division



import matplotlib
#matplotlib.use('Agg')  # questo fa funzionare matplotlib senza interfaccia grafica (es su un server... )

import numpy as np
 
from gpdswpy.binning import ixpeHistogram1d,ixpeHistogram2d
from gpdswpy.logging_ import logger
from gpdswpy.dqm import ixpeDqmTask, ixpeDqmArgumentParser
from gpdswpy.fitting import fit_gaussian_iterative, fit_histogram,  fit_modulation_curve
from gpdswpy.filtering import full_selection_cut_string, energy_cut, cut_logical_and
from gpdswpy.modeling import ixpeFe55
from gpdswpy.matplotlib_ import plt
from gpdswpy.tasks.pha_trend import pha_trend
from scipy.interpolate import InterpolatedUnivariateSpline
from gpdswpy.run import ixpeRunList

import glob


__description__ = 'PHA spectrum'
parser = ixpeDqmArgumentParser(description=__description__)


parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize the pulse height distribution area to 1')
parser.add_argument('--fit', action='store_true', default=False,
                    help='fit the pulse height distribution')
parser.add_argument('--fit-model', type=str, help='fit model',
                    default='gauss', choices=['gauss', 'Fe55'])
parser.add_argument('--fit-min', type=float, nargs='?', default=0.,
                    help='fit lower edge for Fe55 model')
parser.add_argument('--fit-max', type=float, nargs='?', default=40000.,
                    help='fit upper edge for Fe55 model')
parser.add_argument('--nsigma', type=float, nargs='?', default=1.5,
                    help='fitting range in number of sigma for gauss model')
parser.add_argument('--seconds-per-bin', type=float, nargs='?', default=600,
                    help='number of seconds in one temporal bin')
parser.add_argument('--correct-peak-drift', action='store_true',
                    default=False,
                    help='Correct spectrum for peak drift')

parser.add_argument('--degrees', action='store_true', default = False,
                    help='plot the modulation curve with angles in degrees')

parser.add_pha_options()
#parser.set_defaults(pha_expr='TRK_PI')
#parser.set_defaults(pha_expr='TRK_PHA')
parser.set_defaults(pha_expr='PHA_EQ')

parser.add_cut_options()

# ((abs(TRK_BARX) < 7.000) && (abs(TRK_BARY) < 7.000)) && ( (NUM_CLU > 0) && (LIVETIME > 15))

### PARAMETRI....
cut_sigma=2
cut_base='((abs(TRK_BARX) < 7.000) && (abs(TRK_BARY) < 7.000)) && ( (NUM_CLU > 0) && (LIVETIME > 15) && (TRK_SIZE > 0) )'
#cut_base='( TRK_BARX >-1) && ( TRK_BARX <-0.2) &&  ( TRK_BARY >0.4) && ( TRK_BARY <0.8)   && ( (NUM_CLU > 0) && (LIVETIME > 15)  )'





def peak_cut(model):
    """
    """
    peak = model.parameter_value('Peak')
    sigma = model.parameter_value('Sigma')
    #num_cut_sigma = kwargs.get('cut_sigma')
    num_cut_sigma = cut_sigma
    
    plt.axvline(peak - num_cut_sigma * sigma, label='selection region')
    plt.axvline(peak + num_cut_sigma * sigma)
    return energy_cut(peak, sigma, nsigma=num_cut_sigma)


def computeUQmufinal(Q0,U0,sigmaQ0,sigmaU0,Q90,U90,sigmaQ90,sigmaU90):
    Qsp=(Q0+Q90)/2.
    Usp=(U0+U90)/2.

    sigmaQsp=0.5*(sigmaQ0**2+sigmaQ90)**0.5
    sigmaUsp=0.5*(sigmaU0**2+sigmaU90)**0.5

    
    
    muSp=( (Qsp**2)+(Usp**2) )**0.5
    muSp_err=np.sqrt(  ((Qsp*sigmaQsp)**2)+((Usp*sigmaUsp)**2) )/muSp
    
    
    Qs=(Q0-Q90)/2.
    Us=(U0-+U90)/2.
    mus=( (Qs**2)+(Us**2) )**0.5 
    mus_err=np.sqrt(  ((Qs*sigmaQsp)**2)+((Us*sigmaUsp)**2) )/mus
   


    
    print("Qspuria=",Qsp," Uspuria=",Usp,"  ======>>>> ",muSp," +-",muSp_err)
    print("Qsource=",Qs," Usource=",Us,"  ======>>>> ",mus," +-",  mus_err )
   


def compute_QUtot(phi):
    print ("phi=",phi)
    q=2.*np.cos(2.*phi) # phi in rad!!!
    u=2.*np.sin(2.*phi) # phi in rad!!!


    nev=float(len(q))
    Q=np.sum(q)/float(len(q))
    U=np.sum(u)/float(len(u))

    sigmaQ=np.sqrt((2-Q**2)/(nev-1))
    sigmaU=np.sqrt((2-U**2)/(nev-1))
   
    
    mu=(Q**2+U**2)**0.5
    mu_err=np.sqrt(  ((Q*sigmaQ)**2)+((U*sigmaU)**2) )/mu
    
    print("Q=",Q,"U=",U, "n=",float(len(u))," mu=",mu," +-",mu_err," ( ",mu_err/mu,"% )"  ) 

    
    return Q,U,sigmaQ,sigmaU

    
def makeUQmaps(x,y,phi):

    q=2.*np.cos(2.*phi) # phi in rad!!!
    u=2.*np.sin(2.*phi) # phi in rad!!!

    nBins=100
    
    countsMap, xedges, yedges= np.histogram2d(x,y,bins=[nBins,nBins],range=[[-7.5,7.5],[-7.5,7.5]],density=False)
    QMap, xedges, yedges= np.histogram2d(x,y, weights=q,bins=[nBins, nBins],range=[[-7.5,7.5],[-7.5,7.5]],density=False)
    UMap, xedges, yedges= np.histogram2d(x,y, weights=u,bins=[nBins,nBins],range=[[-7.5,7.5],[-7.5,7.5]],density=False)
   
    QMap=QMap/countsMap
    UMap=UMap/countsMap

    QErr_map= np.sqrt((2-np.square(QMap))/(countsMap-1) )
    UErr_map= np.sqrt((2-np.square(UMap))/(countsMap-1) )
    

    return QMap,UMap,countsMap, QErr_map,UErr_map
     
class find_photoPeak(ixpeDqmTask):
    """
    """
    def find_ecut(self,**kwargs):
        """
        """
        pha_min = kwargs.get('min_pha')
        pha_max = kwargs.get('max_pha')
        assert(pha_max > pha_min)
        pha_bins = kwargs.get('pha_bins')
        assert(pha_bins > 0)
        overwrite = kwargs.get('overwrite')
        pha_expr = kwargs.get('pha_expr')
        print("!!! pha expr=",pha_expr)
        pha_binning = np.linspace(pha_min, pha_max, pha_bins + 1)
        pha_title='Pulse height [ADC counts]'
        if (pha_expr == 'TRK_PI'):
            pha_title = 'Pulse Invariant [norm. ADC counts]'
        hist = ixpeHistogram1d(pha_binning, xtitle=pha_title)                                      
        cut=cut_base
        logger.info('Full selection cut (for pha_spectrum) : %s' % cut)
               
        logger.info('Filling the histograms...')
        pha = self.run_list.values(pha_expr, cut) #!!!! qua fa il cut!!! 
        hist.fill(pha)                            
        print ("n_ev rimasti = ",len(pha))
        self.add_plot('pha_spectrum', hist, figure_name='pha_spectrum',  stat_box_position=None, label=kwargs.get('label'),  save=False)

        # cerco il bin center corrispondente al max dell'istogramma. Per fittare intorno al picco.
        # il fit su tutto il range in alcuni casi non converge (es se ho un doppio picco )
        index_max=np.where(hist.bin_weights==hist.max_val())[0][0]
        x_max= hist.bin_centers[0][index_max]                    
        deltaX=3.5*x_max*0.1
        print("max index=",index_max," mac center = ",x_max, " detal = ",deltaX)
        nsigma = 1
        
        gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=x_max-deltaX,  xmax=x_max+deltaX, num_sigma_left=nsigma,  num_sigma_right=nsigma, num_iterations=5) # n. iterazioni??

       
        self.add_plot('pha_spectrum_fit', gauss_model  , figure_name='pha_spectrum',    save=False,       display_stat_box=kwargs.get('display_stat_box', True),    position=kwargs.get('position', 'upper left'))
        self.save_figure('pha_spectrum', overwrite=overwrite)

        peak = gauss_model.parameter_value('Peak')
        peak_err = gauss_model.parameter_error('Peak')
        resolution = gauss_model.resolution()
        resolution_err = gauss_model.resolution_error()
        print("peak = ",peak," +- ",peak_err," res fwhm =",resolution," +- ",resolution_err)

        ecut = peak_cut(gauss_model)
        print ("Ecut = ",ecut)
        plt.savefig(kwargs.get('output_folder')+'PHA_spectrum1.png')

        return ecut
         


    

class plotAll_simo(ixpeDqmTask):

    """
    """
    
    def plotAndFitPhi(self,phi,kwargs):
        edge=180
        nbins=360
        ang_binning = np.linspace(-edge, edge, nbins + 1)
        #print ("ang_bnins = ",ang_binning)
        
        hist_phi = ixpeHistogram1d(ang_binning, xtitle='deg')
        hist_phi.fill(np.degrees(phi))
        self.add_plot('hist_phi', hist_phi, figure_name='hist_phi')
        # fitto phi1:
        fit_model1 = fit_modulation_curve(hist_phi, xmin=-edge, xmax=edge, degrees=True,  verbose=kwargs.get('verbose'))
        self.add_plot('modulation_curve',  fit_model1,figure_name='hist_phi', save=False, display_stat_box=kwargs.get('display_stat_box', True), position=kwargs.get('position', 'lower left'))
        plt.savefig(kwargs.get('output_folder')+'modulation_phi1.png')


    
  
    
    def do_run(self,ecutAll, **kwargs):
        """
        """
               
       
               
        ###################################3
        # mappa bary e mappa punto impatto
        # mi serve un histo 2D... 

        
        ecut= ecutAll
        cut2= cut_logical_and(cut_base,ecut)
        #cut2=cut_base       
        print ("cut_final = ",cut2)
       
        x = self.run_list.values('TRK_BARX', cut2)
        y = self.run_list.values('TRK_BARY', cut2)
        phi1= self.run_list.values('TRK_PHI2', cut2)

        Q,U,sigmaQ,sigmaU= compute_QUtot(phi1)
        self.plotAndFitPhi(phi1,kwargs)
        Qmap,Umap,countsMaps,QErr_map,UErr_map=makeUQmaps(x,y,phi1)

        return Q,U,sigmaQ,sigmaU,  Qmap,Umap,countsMaps,QErr_map,UErr_map
       



      
if __name__ == '__main__':
    args = parser.parse_args()
    opts = vars(args)
    opts['file_type'] = 'Lvl1a'


    fAll=glob.glob('/home/maldera/Desktop/eXTP/data/GPD_DATA/35nonPol_long/*/*.fits')
    taskAll = find_photoPeak(*fAll, **opts)
    ecut= taskAll.find_ecut(**opts)

    print("========================>>>>>>>>>>>>>>>> Ecut ALL=",ecut)
    del taskAll
    
    
    f0=glob.glob('/home/maldera/Desktop/eXTP/data/GPD_DATA/35nonPol_long/deg0/*.fits')
    print ("Processing files... : ",f0)
    task0 = plotAll_simo(*f0, **opts)
    Q0,U0,sigmaQ0,sigmaU0,Q0map,U0map,countsMaps0,QErr_map0,UErr_map0=task0.do_run(ecut,**opts)
    


    #f90=glob.glob('/home/maldera/Desktop/eXTP/data/GPD_DATA/35nonPol_long/deg90/911_0000895_data_Eq50.fits')
    f90=glob.glob('/home/maldera/Desktop/eXTP/data/GPD_DATA/35nonPol_long/deg90/*.fits')
    print ("Processing files... : ",f90)
    task90 = plotAll_simo(*f90, **opts)
    Q90,U90,sigmaQ90,sigmaU90 ,Q90map,U90map,countsMaps90,QErr_map90,UErr_map90 =task90.do_run(ecut,**opts)


    computeUQmufinal(Q0,U0,sigmaQ0,sigmaU0,Q90,U90,sigmaQ90,sigmaU90)  


    #mappa spuria:
    mapQsp=(Q0map+Q90map)/2.
    mapUsp=(U0map+U90map)/2.

    sigmaQsp=0.5*np.sqrt((np.square(QErr_map0)+np.square(QErr_map90)))
    sigmaUsp=0.5*np.sqrt((np.square(UErr_map0)+np.square(UErr_map90)))
    mapSp=np.sqrt(np.square(mapQsp)+np.square(mapUsp))

    mapSp_err=np.sqrt(  (np.square(mapQsp*sigmaQsp))+(np.square(mapUsp*sigmaUsp)) )/mapSp
    
    

    fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(11,7) )
    #fig1, ax1 = plt.subplots()
    #mapSp=   mapSp.T
    im1=ax1[0].imshow(mapSp, interpolation='nearest', origin='lower',  extent=[-7.5,7.5 , -7.5, 7.5] ) 
    fig1.colorbar(im1,ax=ax1[0])           

    allSp=mapSp.flatten()
    mu_i, bins_mu = np.histogram(allSp,  bins =100 , range = (0,1) )
    #fig, h_mu = plt.subplots()
    ax1[1].hist(bins_mu[:-1], bins = bins_mu, weights = mu_i, histtype = 'step',label="dist mu spuria")
    plt.legend()

    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(11,7) )
    im2=ax2[0].imshow(mapSp_err, interpolation='nearest', origin='lower',  extent=[-7.5,7.5 , -7.5, 7.5] ) 
    fig2.colorbar(im2,ax=ax2[0])           
    allSp_err=mapSp_err.flatten()
    muErr_i, bins_muErr = np.histogram(allSp_err,  bins =100,range=[0,1] )
    #fig, h_mu = plt.subplots()
    ax2[1].hist(bins_muErr[:-1], bins = bins_muErr, weights = muErr_i, histtype = 'step',label="Errore dist mu spuria")
    plt.legend()
    
    
    #if args.__dict__['show']:
    plt.show()
