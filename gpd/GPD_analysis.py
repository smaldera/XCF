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
import ast
from matplotlib.patches import Polygon
from gpdswpy.binning import ixpeHistogram1d,ixpeHistogram2d
from gpdswpy.logging_ import logger
from gpdswpy.dqm import ixpeDqmTask, ixpeDqmArgumentParser
from gpdswpy.fitting import fit_gaussian_iterative, fit_histogram,  fit_modulation_curve
from gpdswpy.filtering import full_selection_cut_string, energy_cut, cut_logical_and
from gpdswpy.modeling import ixpeFe55
from gpdswpy.matplotlib_ import plt
from gpdswpy.tasks.pha_trend import pha_trend
from gpdswpy.stokes import *
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

parser.add_argument('--cut', type=str, help='cut to be applied',
                    default=None)
parser.add_argument('--cut-type', type=str, help='type of cut to be applied',
                    default=None,choices=['custom','rectangular'])
parser.add_argument('--cut-file', type=str, help='file containing the cut to be applied', default=None)
parser.add_argument('--suffix', type=str, help='suffix to save the data', default=None)
parser.add_argument('--polcoord', type=str, help='array coord for polygon cut', default=None)
parser.add_argument('--cut-sigma', type=float, default=3, help='sigma cut')
parser.add_argument('--mod-bins', type=int, default=360, help='modulation bins')
parser.add_argument('--map-bins', type=int, default=200, help='map bins')
parser.add_argument('--tbins', type=int, default=10, help='time bins')
parser.add_argument('--save-phi', action='store_true',default=False,help='save phi arrays')
parser.add_argument('--save-pha', action='store_true',default=False,help='save pha array')
parser.add_argument('--no-ecut', action='store_true',default=False,help='do not perform an energy cut')


parser.add_pha_options()
#parser.set_defaults(pha_expr='TRK_PI')
#parser.set_defaults(pha_expr='TRK_PHA')
parser.set_defaults(pha_expr='PHA_EQ')

parser.add_cut_options()



### PARAMETRI....
cut_sigma=parser.parse_args().cut_sigma
cut_base='( (abs(TRK_BARX) < 7.000) && (abs(TRK_BARY) < 7.000) ) && ( (NUM_CLU > 0) && (TRK_SIZE > 0) )' #&& (LIVETIME > 15) )'# && (TRK_PI<30000) && (TRK_PI>22000)  )'
#cut_base='( TRK_BARX >-1) && ( TRK_BARX <-0.2) &&  ( TRK_BARY >0.4) && ( TRK_BARY <0.8)   && ( (NUM_CLU > 0) && (LIVETIME > 15)  )'

cut_gem = '( (TRK_BARX > 4.960) + (TRK_BARX < 4.799) + (TRK_BARY > -2.559) + (TRK_BARY < -2.641) )'

cut_base = cut_logical_and(cut_base,cut_gem)


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
       A function that for a given spectrum
       model (a Gaussian model for example)
       returns an energy cut string
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
    def get_cut(self, **kwargs):
        '''
           function that returns the 
           cut passed as argument
        '''
        cut = kwargs.get('cut')
        return cut

    def merge_cut(self, new_cut, **kwargs):
        '''
           Function that can merge two cuts using the 
           cut_logical_and function. If the new cut is
           not given, the returned cut is the cut base
        '''
        if new_cut is not None:
            if kwargs.get('cut') is not None:
                cut = cut_logical_and(new_cut,kwargs.get('cut'))
            else:
                cut = new_cut
        else:
            if kwargs.get('cut') is not None:
                cut = kwarsg.get('cut')
            else:
                cut = cut_base
        print(f'Cut in action = {cut}')
        return cut
    
    def gauss_model_fit(self, hist, name, figure_name, new_show=True, **kwargs):
        index_max=numpy.where(hist.bin_weights==hist.max_val())[0][0]
        x_max= hist.bin_centers[0][index_max]                    
        deltaX=3.5*x_max*0.1
        print("max index = ",index_max,", max center = ",x_max, ", deltaX = ",deltaX)
        nsigma = kwargs.get('nsigma')
        
        if kwargs.get('fit_min')!=0 or kwargs.get('fit_max')!=40000.0:
            gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=kwargs.get('fit_min'), xmax=kwargs.get('fit_max'), num_sigma_left=nsigma, num_sigma_right=nsigma, num_iterations=10)
        else:
            gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=x_max-deltaX, xmax=x_max+deltaX, num_sigma_left=nsigma, num_sigma_right=nsigma, num_iterations=2)

        if new_show:
            self.add_plot(name, gauss_model, figure_name=figure_name, save=False, display_stat_box=kwargs.get('display_stat_box', True), position=kwargs.get('position', 'upper right'))
        peak = gauss_model.parameter_value('Peak')
        peak_err = gauss_model.parameter_error('Peak')
        resolution = gauss_model.resolution()
        resolution_err = gauss_model.resolution_error()
        print('Fit parameters:')
        print("peak = ",peak," +- ",peak_err,"\nres fwhm =",resolution," +- ",resolution_err)
        return gauss_model

    
    def pha_spectrum_hist(self, merge_cut, extra_cut, **kwargs):
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
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

    def pha_spectrum_plot(self, figure_name, merge_cut, extra_cut, suf, **kwargs):
        print(f'####\nPHA spectrum')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        hist = self.pha_spectrum_hist(merge_cut=merge_cut, extra_cut=cut, **kwargs)
        self.add_plot('pha_spectrum', hist, figure_name=figure_name,  stat_box_position=None, label=kwargs.get('label'),  save=False)

        if kwargs.get('fit')==True:
            gauss_model = self.gauss_model_fit(hist=hist, name='pha_spectrum', figure_name=figure_name, **kwargs)
            if kwargs.get('no_ecut')==False:
                ecut = cut_logical_and(cut,peak_cut(gauss_model))
                print ("Energuy cut = ", ecut)
                if kwargs.get('output_folder')!=None:
                    if suf is not None:
                        plt.savefig(kwargs.get('output_folder')+f'PHA_spectrum1_{suf}.png')
                    else:
                        plt.savefig(kwargs.get('output_folder')+f'PHA_spectrum1.png')
                return ecut, gauss_model
            if kwargs.get('no_ecut')==True:
                ecut = cut
                print ("Energuy cut = ", ecut)
                if kwargs.get('output_folder')!=None:
                    if suf is not None:
                        plt.savefig(kwargs.get('output_folder')+f'PHA_spectrum1_noecut_{suf}.png')
                    else:
                        plt.savefig(kwargs.get('output_folder')+f'PHA_noecut_spectrum1.png')
                return ecut, gauss_model
        else:
            ecut_ = f'( (TRK_PI < {kwargs.get("max_pha")}) && (TRK_PI > {kwargs.get("min_pha")}) )'
            ecut = cut#cut_logical_and(cut,ecut_)
            gauss_model = None
            if kwargs.get('output_folder')!=None:
                if suf is not None:
                    plt.savefig(kwargs.get('output_folder')+f'PHA_spectrum1_{suf}.png')
                else:
                    plt.savefig(kwargs.get('output_folder')+f'PHA_spectrum1.png')
            return ecut, gauss_model
            
        
        #self.save_figure('pha_spectrum', overwrite=kwargs.get('overwrite'))
        
        
                
    def projections(self, coord, expression, figure_name, merge_cut, extra_cut, cut_edges, suf, **kwargs):
        print(f'#####\nProjection for {expression} {coord}')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        coordinate = self.run_list.values(f'{expression}{coord}', cut)
        coord_min = -8.
        coord_max = 8.
        nside = 320
        edges = numpy.linspace(coord_min, coord_max, nside+1)
        hist_proj = ixpeHistogram1d(edges, xtitle=f'{expression}{coord} mm')
        hist_proj.fill(coordinate)
        self.add_plot(f'{expression}{coord}_proj', hist_proj, figure_name=figure_name)
        if cut_edges is not None:
            for x in cut_edges:
                plt.axvline(x=x,linestyle='--')
        if kwargs.get('output_folder')!=None:
            if suf is not None:
                plt.savefig(kwargs.get('output_folder')+f'{expression}{coord}_proj_{suf}.png')
            else:
                plt.savefig(kwargs.get('output_folder')+f'{expression}{coord}_proj.png')
        return 
            
    def multiple_projections(self, coord, expression, figure_name, merge_cut, extra_cut, coord_slice, expression_slice, suf, **kwargs):
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        coordinate = self.run_list.values(f'{expression}{coord}', cut)
        coord_min = -8.
        coord_max = 8.
        nside = 320
        edges = numpy.linspace(coord_min, coord_max, nside+1)
        hist_proj = ixpeHistogram1d(edges, xtitle=f'{expression}{coord} mm')
        hist_proj.fill(coordinate)
        self.add_plot(f'{expression}{coord}_proj', hist_proj, figure_name=figure_name)
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+f'{expression}{coord}_proj{suf}.png')
        


    def map(self, expression, merge_cut, extra_cut, map_title, shape, cut_edges, suf, **kwargs):
        print(f'####\nMap for {expression}')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        x = self.run_list.values(f'{expression}X', cut)
        y = self.run_list.values(f'{expression}Y', cut)
        x_min=-8
        x_max=8
        y_min=-8
        y_max=8
        nside=kwargs.get('map_bins')
        x_edges = numpy.linspace(x_min, x_max, nside +1)
        y_edges = numpy.linspace(y_min, y_max, nside +1)
        hist_map = ixpeHistogram2d(x_edges, y_edges,  xtitle='x [mm]', ytitle='y [mm]')
        hist_map.fill(x, y)
        self.add_plot(f'{expression} map', hist_map, figure_name=map_title)
        if shape=='rectangular':
            for x in cut_edges[0]:
                plt.axvline(x=x,linestyle='--')
            for y in cut_edges[1]:
                plt.axhline(y=y,linestyle='--')
        if shape=='circle':
            circle = plt.Circle((cut_edges[0], cut_edges[1]), cut_edges[2], color='cyan', linewidth=2, linestyle='--',fill=False)
            plt.gca().add_patch(circle)
        if shape=='polygon':
            polygon_ = Polygon(xy=list(zip(cut_edges[0], cut_edges[1])), closed=True, color='cyan', linewidth=2, linestyle='--', fill=False)
            plt.gca().add_patch(polygon_)
        if kwargs.get('output_folder')!=None:
            if suf is not None:
                plt.savefig(kwargs.get('output_folder')+f'bary_map_{suf}.png')
            else:
                plt.savefig(kwargs.get('output_folder')+f'bary_map.png')
            

    def modulation(self, phi, modulation_title, merge_cut, extra_cut, suf, **kwargs):
        print(f'####\nModulation plot for phi{phi}')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        phi_values= self.run_list.values(f'numpy.degrees(TRK_PHI{phi})', cut)
        
        edge=180
        nbins = kwargs.get('mod_bins')
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
            if suf is not None:
                plt.savefig(kwargs.get('output_folder')+f'modulation_phi{phi}_{suf}.png')
            else:
                plt.savefig(kwargs.get('output_folder')+f'modulation_phi{phi}.png')
            
        return phase, phase_err, modulation, modulation_err, chi2, cut#, hist_phi.bin_edges, hist_phi.bin_centers, hist_phi.bin_weights

    def save_phi(self,phi1,phi2,suf,**kwargs):
        if kwargs.get('output_folder') is not None:
            if suf is not None:
                numpy.savez(kwargs.get('output_folder')+f'phi_{suf}.npz', array1=phi1, array2=phi2)
            else:
                numpy.savez(kwargs.get('output_folder')+f'phi.npz', array1=phi1, array2=phi2)

    def save_pha(self, merge_cut, extra_cut, suf, **kwargs):
        pha_expr = kwargs.get('pha_expr')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        pha = self.run_list.values(pha_expr, cut)
        if kwargs.get('output_folder') is not None:
            if suf is not None:
                numpy.savez(kwargs.get('output_folder')+f'pha_{suf}.npz', pha)
            else:
                numpy.savez(kwargs.get('output_folder')+f'pha.npz', pha)

    def cut_string(self,cut,symbol):
        cut_ = cut.split(symbol)
        return cut_

    def cut_string_coords(self,cut):
        x_bar, y_bar = [], []
        x_abs, y_abs = [], []
        for c in cut:
            c_ = c.split(' ')
            if c_[1] == f'TRK_BARX':
                x_bar.append(float(c_[3]))
            if c_[1] == f'TRK_BARY':
                y_bar.append(float(c_[3]))
            if c_[1] == f'TRK_ABSX':
                x_abs.append(float(c_[3]))
            if c_[1] == f'TRK_ABSY':
                y_abs.append(float(c_[3]))
        return [x_bar,y_bar], [x_abs,y_abs]
    
    def cut_string_rect(self,cut,expression):
        cut_ = self.cut_string(cut,'(')
        new = []
        for i in range(len(cut_)):
            new.append(cut_[i].split(')'))
        new_ = []
        for i in range(len(new)):
            for a in new[i]:
                new_.append(a)
        x,y = [],[]

        if expression=='TRK_BAR':
            for e in new_:
                if e.split('>')[0] == 'TRK_BARX':
                    x.append(float(e.split('>')[1]))
                if e.split('>')[0] =='TRK_BARY':
                    y.append(float(e.split('>')[1]))
                if e.split('<')[0] == 'TRK_BARX':
                    x.append(float(e.split('<')[1]))
                if e.split('<')[0] =='TRK_BARY':
                    y.append(float(e.split('<')[1]))
        if expression=='TRK_ABS':
            for e in new_:
                if e.split('>')[0] == 'TRK_ABSX':
                    x.append(float(e.split('>')[1]))
                if e.split('>')[0] =='TRK_ABSY':
                    y.append(float(e.split('>')[1]))
                if e.split('<')[0] == 'TRK_ABSX':
                    x.append(float(e.split('<')[1]))
                if e.split('<')[0] =='TRK_ABSY':
                    y.append(float(e.split('<')[1]))
        return [x,y]

    def cut_string_circ(self,cut,expression):
        '''
           
        '''
        cut_ = self.cut_string(cut,'(')
        new = []
        for i in range(len(cut_)):
            new.append(cut_[i].split(')'))
        new_ = []
        for i in range(len(new)):
            for a in new[i]:
                new_.append(a)
        x = 0
        y = 0
        r = 0
        if expression=='TRK_BAR':
            for e in new_:
                if e.split(' ')[0]=='TRK_BARX':
                    x = float(e.split(' ')[2])
                if e.split(' ')[0]=='TRK_BARY':
                    y = float(e.split(' ')[2])
                if len(e.split(' '))>2:
                    if e.split(' ')[1]=='<':
                        print(e.split(' '))
                        rr = e.split(' ')[2]
                        r = float(rr.split('**')[0])
        if expression=='TRK_ABS':
            for e in new_:
                if e.split(' ')[0]=='TRK_ABSX':
                    x = float(e.split(' ')[2])
                if e.split(' ')[0]=='TRK_ABSY':
                    y = float(e.split(' ')[2])
                if len(e.split(' '))>2:
                    if e.split(' ')[1]=='<':
                        print(e.split(' '))
                        rr = e.split(' ')[2]
                        r = float(rr.split('**')[0])

        return x, y, r

    def get_phi_array(self, phi, merge_cut, extra_cut, **kwargs):
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        print(f'\nSTOKES EVENTS CUT = {cut}\n')
        phi_values= self.run_list.values(f'TRK_PHI{phi}', cut)
        return phi_values
    
    def STOKES(self, phi, **kwargs):
        I_ = I(phi)
        Q_ = Q(phi)
        dQ_ = dQ(phi)
        U_ = U(phi)
        dU_ = dU(phi)
        return I_, Q_, dQ_, U_, dQ_

    def STOKES_NORM(self, phi, **kwargs):
        I, Q, dQ, U, dQ = self.STOKES(phi,**kwargs)
        QN = q_norm(Q,I)
        dQN = dq_norm(QN,I)
        UN = u_norm(U,I)
        dUN = du_norm(UN,I)
        return I, QN, dQN, UN, dUN

    def peak_vs_time(self, figure_name, merge_cut, extra_cut, suf, **kwargs):
        print(f'#####\nTIME')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        time = self.run_list.values(f'TIME', cut)
        time_min = numpy.min(time)
        time_max = numpy.max(time)
        tbins = kwargs.get('tbins')
        tedges = numpy.linspace(time_min, time_max, tbins+1)
        peak_t, peak_err_t, res_t, res_err_t = [], [], [], []
        cut_binned = []
        for i in range(len(tedges)-1):
            cut_time = f"( (TIME > {tedges[i]}) && (TIME < {tedges[i+1]}) )"
            cut_time_bin = cut_logical_and(cut,cut_time)
            cut_binned.append(cut_time_bin)
            hist = self.pha_spectrum_hist(merge_cut=merge_cut, extra_cut=cut_time_bin, **kwargs)
            try:
                gauss_model = self.gauss_model_fit(hist=hist, name='pha_spectrum', figure_name=figure_name, new_show=False,**kwargs)
                peak_t.append(gauss_model.parameter_value('Peak'))
                peak_err_t.append(gauss_model.parameter_error('Peak'))
                res_t.append(gauss_model.resolution())
                res_err_t.append(gauss_model.resolution_error())
            except RuntimeError:
                print('ERROR!!!!!!!')
                peak_t.append(0.)
                peak_err_t.append(0.)
                res_t.append(0.)
                res_err_t.append(0.)
        #breakpoint()
        time_centers = (tedges[1:]+tedges[:-1])/2
        
        fig_, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(time_centers, peak_t, yerr=peak_err_t, linestyle='-', marker='o', capsize=4)
        ax2.errorbar(time_centers, res_t, yerr=res_err_t, linestyle='-', marker='o', capsize=4)
        ax1.set_ylabel('peak')
        ax2.set_ylabel('resolution')
        ax2.set_xlabel('time [s]')
        ax1.grid(True)
        ax2.grid(True)



        """
           DO RUN !!!
        """

        
    def do_run(self, **kwargs):
        """
        """
        
        if kwargs.get('cut_type')=='rectangular':
            external_cut = self.get_cut(**kwargs)
            print(f'\nExternal cut = {external_cut}\n')
            coord_ = self.cut_string_rect(external_cut,expression='TRK_BAR')
            print(coord_)
            cut_suf = kwargs.get('suffix')
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape='rectangular', cut_edges=coord_, suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=coord_[0], suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=coord_[1], suf=cut_suf, **kwargs)
            cut_base_new = cut_logical_and(cut_base,external_cut)
            n_physical=self.run_list.num_events(cut_base_new)
            
        if kwargs.get('cut_type')=='circle':
            external_cut = self.get_cut(**kwargs)
            print(f'\nExternal cut = {external_cut}\n')
            x,y,r = self.cut_string_circ(external_cut,expression='TRK_BAR')
            coord_ = [x,y,r]
            cut_suf = kwargs.get('suffix')
            print(coord_)
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape='circle', cut_edges=coord_, suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=[coord_[0]-coord_[2],coord_[0]+coord_[2]], suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=[coord_[1]-coord_[2],coord_[1]+coord_[2]], suf=cut_suf, **kwargs)
            cut_base_new = cut_logical_and(cut_base,external_cut)
            n_physical=self.run_list.num_events(cut_base_new)

        if kwargs.get('cut_type')=='polygon':
            external_cut = self.get_cut(**kwargs)
            print(f'\nExternal cut = {external_cut}\n')
            pol_coord = kwargs.get('polcoord')
            x_pol = ast.literal_eval(pol_coord)[0]
            y_pol = ast.literal_eval(pol_coord)[1]
            cut_suf = kwargs.get('suffix')
            print(x_pol, y_pol)
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape='polygon', cut_edges=[x_pol,y_pol], suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=x_pol, suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=y_pol, suf=cut_suf, **kwargs)
            cut_base_new = cut_logical_and(cut_base,external_cut)
            n_physical=self.run_list.num_events(cut_base_new)
            
        if kwargs.get('cut_type')==None:
            cut_suf = kwargs.get('suffix')
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape=None, cut_edges=None, suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=None, suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=None, suf=cut_suf, **kwargs)
            n_physical=self.run_list.num_events(cut_base)
        
        ecut, gauss_model = self.pha_spectrum_plot(figure_name='pha_spectrum', merge_cut = True, extra_cut = cut_base, suf=cut_suf, **kwargs) 
        
        track_size_cut='(TRK_SIZE > 0)'
        cut2= cut_logical_and(ecut,track_size_cut)

        '''
        if gauss_model is not None:

            n_physical=self.run_list.num_events(cut_base)
            n_ecut=self.run_list.num_events(cut2)
        
            ecut_efficiency = n_ecut/n_physical
            quantile = min(0.8/ecut_efficiency, 1.)
            expr = 'TRK_M2L/TRK_M2T'
            print('\nQUANTILE CUT\n')
            print(f'efficiency = {ecut_efficiency}, n cut = {n_ecut}, n physical = {n_physical}\n')
            print(f'quantile = {quantile}, expression = {expr}\n')

            self.add_plot('moments ratio', gauss_model, figure_name='moments ratio')
            min_mom_ratio = find_quantile(self.run_list, quantile, expr, cut2)
            if kwargs.get('output_folder')!=None:
                plt.savefig(kwargs.get('output_folder')+'dist_ratioLW.png')
            mom_ratio_cut = '%s > %.4f' % (expr, min_mom_ratio)
            cut_final=cut_logical_and(cut2,mom_ratio_cut)

            print ("\ncut_final = ",cut_final,'\n')

        else:
            cut_final = cut2
        '''
        cut_final = cut2

        self.peak_vs_time(figure_name='peak_vs_time', merge_cut=True, extra_cut=cut_final, suf=None, **kwargs)
        
        ###################################
        self.map(expression='TRK_BAR', merge_cut=True,  extra_cut=cut_final, map_title='barycenter map cut', shape=None, cut_edges=None, suf='cut_'+cut_suf, **kwargs)
        self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj cut', merge_cut=True, extra_cut=cut_final, cut_edges=None, suf='cut_'+cut_suf, **kwargs)
        self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj cut', merge_cut=True, extra_cut=cut_final, cut_edges=None, suf='cut_'+cut_suf, **kwargs)
        
        
        ###################################
        # istogramma ph1
        phase_1, phase_err_1, modulation_1, modulation_err_1, chi2_1, cut_1 = self.modulation(phi=1, modulation_title='modulation phi 1', merge_cut=True, extra_cut=cut_final, suf=cut_suf, **kwargs)
        phi_1_arr = self.get_phi_array(phi=1, merge_cut=True, extra_cut=cut_final, **kwargs)
        I_1, QN_1, dQN_1, UN_1, dUN_1 = self.STOKES_NORM(phi_1_arr, **kwargs)
        I_1, Q_1, dQ_1, U_1, dU_1 = self.STOKES(phi_1_arr, **kwargs)
        mu_1, mu_err_1, phase_st_1, phase_st_err_1 = polarization(I_1,Q_1,U_1,**kwargs)
        phase_st_1 = numpy.degrees(phase_st_1)
        phase_st_err_1 = numpy.degrees(phase_st_err_1)
        
        # istogramma ph2
        phase_2, phase_err_2, modulation_2, modulation_err_2, chi2_2, cut_2 = self.modulation(phi=2, modulation_title='modulation phi 2', merge_cut=True, extra_cut=cut_final, suf=cut_suf, **kwargs)
        phi_2_arr = self.get_phi_array(phi=2, merge_cut=True, extra_cut=cut_final, **kwargs)
        I_2, QN_2, dQN_2, UN_2, dUN_2 = self.STOKES_NORM(phi_2_arr, **kwargs)
        I_2, Q_2, dQ_2, U_2, dU_2 = self.STOKES(phi_2_arr, **kwargs)
        mu_2, mu_err_2, phase_st_2, phase_st_err_2 = polarization(I_2,Q_2,U_2,**kwargs)
        phase_st_2 = numpy.degrees(phase_st_2)
        phase_st_err_2 = numpy.degrees(phase_st_err_2)
        
        if kwargs.get('save_phi') is not None:
            self.save_phi(phi_1_arr,phi_2_arr,suf=cut_suf,**kwargs)
            
        if kwargs.get('save_pha') is not None:
            self.save_pha(merge_cut=True, extra_cut=cut_base, suf=cut_suf, **kwargs)
        
        print(self.run_list.num_events(cut_final), len(phi_1_arr), len(phi_2_arr))

        #self.pha_spectrum_plot(figure_name='pha_spectrum_cut', energy_cut=False, merge_cut = True, extra_cut = cut_final, suf='cut_'+cut_suf, **kwargs)

        if gauss_model is not None:
            peak = gauss_model.parameter_value('Peak')
            peak_err = gauss_model.parameter_error('Peak')
            resolution = gauss_model.resolution()
            resolution_err = gauss_model.resolution_error()

        N_EVENTS_cut = self.run_list.num_events(cut_final)
        print('\n\n**************************************************************************************************\n')
        print(f'N events after all the cuts = {N_EVENTS_cut}')
        if gauss_model is not None:
            print('SPECTRUM FIT')
            print(f'peak = {round(peak,2)} +- {round(peak_err,2)}')
            print(f'resolution = {round(resolution,3)} +- {round(resolution_err,3)}')
        print(f'modulation_1 = {round(modulation_1*100.,2)} +- {round(modulation_err_1*100.,2)}')
        print(f'modulation_2 = {round(modulation_2*100.,2)} +- {round(modulation_err_2*100.,2)}')
        print(f'modulation_1 STOKES = {round(mu_1*100.,2)} +- {round(mu_err_1*100.,2)}')
        print(f'QN_1 = {round(QN_1*100.,2)} +- {round(dQN_1*100.,2)}')
        print(f'UN_1 = {round(UN_1*100.,2)} +- {round(dUN_1*100.,2)}')
        print(f'modulation_2 STOKES = {round(mu_2*100.,2)} +- {round(mu_err_1*100.,2)}')
        print(f'QN_2 = {round(QN_2*100.,2)} +- {round(dQN_2*100.,2)}')
        print(f'UN_2 = {round(UN_2*100.,2)} +- {round(dUN_2*100.,2)}')
        print('\n**************************************************************************************************\n\n')
        RUN_ID_ =  self.run_list.measurements_id()
        RUN_ID = RUN_ID_.split(' - ')[0]
        
        #results = [peak, peak_err, resolution, resolution_err, phase_1, phase_err_1, modulation_1, modulation_err_1, chi2_1, phase_2, phase_err_2, modulation_2, modulation_err_2, chi2_2, cut_final,  kwargs.get('cut_type'), N_EVENTS_cut, RUN_ID, mu_1, mu_err_1, phase_st_1, phase_st_err_1, mu_2, mu_err_2, phase_st_2, phase_st_err_2]

        if gauss_model is None:
            peak = 0
            peak_err = 0
            resolution = 0
            resolution_err = 0
        
        results = {'RUN_ID': RUN_ID, 'N_EVENTS': N_EVENTS_cut, 'CUT': cut_final, 'CUT_TYPE':  kwargs.get('cut_type'),
                   'PEAK': [peak, peak_err], 'RESOLUTION': [resolution, resolution_err],
                   'MODULATION_1': [modulation_1, modulation_err_1], 'PHASE_1': [phase_1, phase_err_1], 'CHI2_1': chi2_1,
                   'MODULATION_2': [modulation_2, modulation_err_2], 'PHASE_2': [phase_2, phase_err_2], 'CHI2_2': chi2_2,
                   'MU_STOKES_1': [mu_1, mu_err_1], 'PHASE_STOKES_1': [phase_st_1, phase_st_err_1],
                   'MU_STOKES_2': [mu_2, mu_err_2], 'PHASE_STOKES_2': [phase_st_2, phase_st_err_2],
                   'QN_1': [QN_1, dQN_1], 'UN_1': [UN_1, dUN_1], 'QN_2': [QN_2, dQN_2], 'UN_2': [UN_2, dUN_2]}

        return results


        
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
