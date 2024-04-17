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


__description__ = 'PHA spectrum'

import matplotlib
#matplotlib.use('Agg')  # questo fa funzionare matplotlib senza interfaccia grafica (es su un server... )

import numpy

from gpdswpy.binning import ixpeHistogram1d,ixpeHistogram2d
from gpdswpy.logging_ import logger
from gpdswpy.dqm import ixpeDqmTask, ixpeDqmArgumentParser
from gpdswpy.fitting import fit_gaussian_iterative, fit_histogram,  fit_modulation_curve
from gpdswpy.filtering import full_selection_cut_string, energy_cut, cut_logical_and
from gpdswpy.modeling import ixpeFe55
from gpdswpy.matplotlib_ import plt
from gpdswpy.tasks.pha_trend import pha_trend
from scipy.interpolate import InterpolatedUnivariateSpline


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



### PARAMETRI....
cut_sigma=3
cut_base='( (abs(TRK_BARX) < 7.000) && (abs(TRK_BARY) < 7.000)) && ( (NUM_CLU > 0) && (LIVETIME > 15)  )'
#cut_base='( TRK_BARX >-1) && ( TRK_BARX <-0.2) &&  ( TRK_BARY >0.4) && ( TRK_BARY <0.8)   && ( (NUM_CLU > 0) && (LIVETIME > 15)  )'


def find_quantile(run, quantile, expr, cut):
    """
    """
    vals = run.values(expr, cut)
    deltabin = 0.001
    _nbinsqt = int(((max(vals) - min(vals)) / deltabin) + 1)
    nbinsqt = min([_nbinsqt, 50000])
    logger.info(f'M2L/M2T histogram N bins = {nbinsqt}')
    binning = numpy.linspace(min(vals), max(vals), nbinsqt)
    hist = ixpeHistogram1d(binning, vals, xtitle=expr)
    q = hist.quantile(1. - quantile)
    hist.plot(stat_box_position='upper right')
    plot_xmax = hist.quantile(0.99)
    plt.xlim(binning[0], plot_xmax)
    plt.ylim(0,hist.max_val()*1.1) #!!!!!!!!    
    plt.axvline(q)
    return q



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






class GPD_analysis(ixpeDqmTask):

    """
    """
    def gauss_model_fit(self, hist, figure_name, **kwargs):
        index_max=numpy.where(hist.bin_weights==hist.max_val())[0][0]
        x_max= hist.bin_centers[0][index_max]                    
        deltaX=3.5*x_max*0.1
        print("max index = ",index_max,", max center = ",x_max, ", deltaX = ",deltaX)
        nsigma = kwargs.get('nsigma')
        
        if kwargs.get('fit_min')!=0 or kwargs.get('fit_max')!=40000.0:
            gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=kwargs.get('fit_min'), xmax=kwargs.get('fit_max'), num_sigma_left=nsigma, num_sigma_right=nsigma, num_iterations=10)
        else:
            gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=x_max-deltaX, xmax=x_max+deltaX, num_sigma_left=nsigma, num_sigma_right=nsigma, num_iterations=2)

        
        self.add_plot('pha_spectrum_fit', gauss_model, figure_name=figure_name, save=False, display_stat_box=kwargs.get('display_stat_box', True), position=kwargs.get('position', 'upper right'))
        peak = gauss_model.parameter_value('Peak')
        peak_err = gauss_model.parameter_error('Peak')
        resolution = gauss_model.resolution()
        resolution_err = gauss_model.resolution_error()
        print('PHA spectrum fit parameters:')
        print("peak = ",peak," +- ",peak_err,"\nres fwhm =",resolution," +- ",resolution_err)
        return gauss_model

    
    def pha_spectrum_hist(self, cut, **kwargs):
        pha_min = kwargs.get('min_pha')
        pha_max = kwargs.get('max_pha')
        assert(pha_max > pha_min)
        
        pha_bins = kwargs.get('pha_bins')
        assert(pha_bins > 0)
                
        pha_expr = kwargs.get('pha_expr')
        print("!!! pha expr=",pha_expr)
        
        pha_binning = numpy.linspace(pha_min, pha_max, pha_bins + 1)
        
        pha_title='Pulse height [ADC counts]'
        
        if (pha_expr == 'TRK_PI'):
            pha_title = 'Pulse Invariant [norm. ADC counts]'
            
        hist = ixpeHistogram1d(pha_binning, xtitle=pha_title)                                        
        logger.info(f'Full selection cut (for pha_spectrum) : {cut}')
       
        
        logger.info('Filling the histograms...')
        pha = self.run_list.values(pha_expr, cut) #!!!! qua fa il lavoro!!! 
        hist.fill(pha)                            
        print(f"Number of events after the cut = {len(pha)}")
        return hist

    def pha_spectrum_plot(self, figure_name, cut, energy_cut, **kwargs):
        hist = self.pha_spectrum_hist(cut=cut, **kwargs)
        self.add_plot('pha_spectrum', hist, figure_name=figure_name,  stat_box_position=None, label=kwargs.get('label'),  save=False)

        if kwargs.get('fit')==True:
            gauss_model = self.gauss_model_fit(hist=hist, figure_name=figure_name, **kwargs)
            if energy_cut==True:
                ecut = peak_cut(gauss_model)
                print ("Energuy cut = ",ecut)
                return ecut, gauss_model
        
        self.save_figure('pha_spectrum', overwrite=kwargs.get('overwrite'))
        
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+'PHA_spectrum1.png')


    def map(self, expression, cut, map_title, **kwargs):
        x = self.run_list.values(f'{expression}X', cut)
        y = self.run_list.values(f'{expression}Y', cut)
        x_min=-8
        x_max=8
        y_min=-8
        y_max=8
        nside=320
        x_edges = numpy.linspace(x_min, x_max, nside +1)
        y_edges = numpy.linspace(y_min, y_max, nside +1)
        hist_map = ixpeHistogram2d(x_edges, y_edges,  xtitle='x [mm]', ytitle='y [mm]')
        hist_map.fill(x, y)
        self.add_plot(map_title, hist_map, figure_name='bary_map')
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+'bary_map.png')
            

    def modulation(self, phi, modulation_title, cut, **kwargs):
        phi_values= self.run_list.values(f'numpy.degrees(TRK_PHI{phi})', cut)
        
        edge=180
        nbins=360
        ang_binning = numpy.linspace(-edge, edge, nbins + 1)
        
        hist_phi = ixpeHistogram1d(ang_binning, xtitle=r'$\Phi$'+str(phi)+' deg')
        hist_phi.fill(phi_values)
        self.add_plot(f'histogramm_phi{phi}', hist_phi, figure_name=modulation_title)
        fit_model = fit_modulation_curve(hist_phi, xmin=-edge, xmax=edge, degrees=True,  verbose=kwargs.get('verbose'))
        self.add_plot('modulation_curve',  fit_model, figure_name=modulation_title, save=False, display_stat_box=kwargs.get('display_stat_box', True), position=kwargs.get('position', 'lower left'))

        phase = fit_model.parameter_value('Phase')
        phase_err = fit_model.parameter_error('Phase')
        modulation = fit_model.parameter_value('Modulation')
        modulation_err = fit_model.parameter_error('Modulation')
        chi2 = fit_model.reduced_chisquare()
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+f'modulation_phi{phi}.png')
            
    def do_run(self, **kwargs):
        """
        """
        ecut, gauss_model = self.pha_spectrum_plot(figure_name='pha_spectrum', cut=cut_base,  energy_cut=True, **kwargs) 

        track_size_cut='(TRK_SIZE > 0)'
        cut2= cut_logical_and(cut_base,ecut,track_size_cut)


        n_physical=self.run_list.num_events(cut_base)
        n_ecut=self.run_list.num_events(cut2)
        
        ecut_efficiency = n_ecut/n_physical
        quantile = min(0.8/ecut_efficiency, 1.)
        expr = 'TRK_M2L/TRK_M2T'
        
        self.add_plot('moments ratio', gauss_model, figure_name='moments ratio')
        min_mom_ratio = find_quantile(self.run_list, quantile, expr, cut2)
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+'dist_ratioLW.png')
        mom_ratio_cut = '%s > %.4f' % (expr, min_mom_ratio)
        cut_final=cut_logical_and(cut2,mom_ratio_cut)

        print ("cut_final = ",cut_final)

        ###################################
        # barycenter map
        self.map(expression='TRK_BAR', cut=cut_final, map_title='barycenter map', **kwargs)
        
        ###################################
        # istogramma ph1
        self.modulation(phi=1, modulation_title='modulation phi 1', cut=cut_final, **kwargs)
        # istogramma ph2
        self.modulation(phi=2, modulation_title='modulation phi 2', cut=cut_final, **kwargs)
        
        ###################################
        # rifaccio istogramma pha con tagli finali per avere la risuluzione!!!

        """
        self.pha_spectrum_plot(figure_name='pha_spectrum', cut=cut_final, fit=True, energy_cut=False, **kwargs)

        

        

        
        ################################################
        # count events:
        n_raw=self.run_list.num_events()
        n_physical=self.run_list.num_events(cut_base)
        n_ecut=self.run_list.num_events(cut2)
        n_final=self.run_list.num_events(cut_final)
        
        print ("n. raw events= ",n_raw)
        print ("n. physical events (bary + livetime+ NUM_CLU ) = ", n_physical)
        print ("n. ecut (physical+pha_spectrum+_tkr_size) ",n_ecut )
        print ("n. final (physical+pha_spectrum+_tkr_size+axis ratio) ",n_final )
        
        print("eff_ecut= n_ecut/physical",float(n_ecut)/float(n_physical))
        
        print("eff= final/physical",float(n_final)/float(n_physical))
        


        #scrivi outfile
        if kwargs.get('output_folder')!=None:
            print("out dir", kwargs.get('output_folder'))
            nomefileout= kwargs.get('output_folder')+'prova_out.txt'
            print("nomefile_out ",nomefileout )

        
            out_string=str(peak2)+' '+str(peak2_err)+' '+str(resolution2)+' '+str(resolution2_err)+' '+str(phase1)+' '+str(phase1_err)+' '+str(modulation1)+' '+str(modulation1_err)+' '+str(chi2_1)+' '+str(phase2)+' '+str(phase2_err)+' '+str(modulation2)+' '+str(modulation2_err)+' '+str(chi2_2)+' '+str(n_raw)+' '+str(n_physical)+'  '+str(n_ecut)+'  '+str(n_final) 
        
            with open(nomefileout, 'w') as miofile:
                miofile = open(nomefileout,'w')
                miofile.write(out_string)
                miofile.close() # !!!!! il file e' bufferizzato, e riempito solo alla chiusura (o chiamando file.flush). messo con with dovrebbe essere chiuso e scritto comunque quando esce dal loop

       """
              
        
if __name__ == '__main__':
    args = parser.parse_args()
    opts = vars(args)
    opts['file_type'] = 'Lvl1a'
      
    task = GPD_analysis(*args.infiles, **opts)
    task.run(**opts)

    #if args.__dict__['show']:
    plt.show()
