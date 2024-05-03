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

parser.add_argument('--cut', type=str, help='cut to be applied',
                    default=None)
parser.add_argument('--cut-type', type=str, help='type of cut to be applied',
                    default=None,choices=['custom','rectangular'])

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
    
    def gauss_model_fit(self, hist, name, figure_name, **kwargs):
        index_max=numpy.where(hist.bin_weights==hist.max_val())[0][0]
        x_max= hist.bin_centers[0][index_max]                    
        deltaX=3.5*x_max*0.1
        print("max index = ",index_max,", max center = ",x_max, ", deltaX = ",deltaX)
        nsigma = kwargs.get('nsigma')
        
        if kwargs.get('fit_min')!=0 or kwargs.get('fit_max')!=40000.0:
            gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=kwargs.get('fit_min'), xmax=kwargs.get('fit_max'), num_sigma_left=nsigma, num_sigma_right=nsigma, num_iterations=10)
        else:
            gauss_model = fit_gaussian_iterative(hist, verbose=kwargs.get('verbose'), xmin=x_max-deltaX, xmax=x_max+deltaX, num_sigma_left=nsigma, num_sigma_right=nsigma, num_iterations=2)

        
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

    def pha_spectrum_plot(self, figure_name, energy_cut, merge_cut, extra_cut, suf, **kwargs):
        print(f'####\nPHA spectrum')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
        hist = self.pha_spectrum_hist(merge_cut=merge_cut, extra_cut=cut, **kwargs)
        self.add_plot('pha_spectrum', hist, figure_name=figure_name,  stat_box_position=None, label=kwargs.get('label'),  save=False)

        if kwargs.get('fit')==True:
            gauss_model = self.gauss_model_fit(hist=hist, name='pha_spectrum', figure_name=figure_name, **kwargs)
            if energy_cut==True:
                ecut = cut_logical_and(cut,peak_cut(gauss_model))
                print ("Energuy cut = ", ecut)
                return ecut, gauss_model
        
        self.save_figure('pha_spectrum', overwrite=kwargs.get('overwrite'))
        
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+f'PHA_spectrum1{suf}.png')

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
            plt.savefig(kwargs.get('output_folder')+f'{expression}{coord}_proj{suf}.png')
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
        nside=320
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
            circle = plt.Circle((cut_edges[0], cut_edges[1]), cut_edges[2], color='cyan', linestyle='--',fill=False)
            plt.gca().add_patch(circle)
        if kwargs.get('output_folder')!=None:
            plt.savefig(kwargs.get('output_folder')+f'bary_map{suf}.png')
            

    def modulation(self, phi, modulation_title, merge_cut, extra_cut, suf, **kwargs):
        print(f'####\nModulation plot for phi{phi}')
        if merge_cut == True:
            cut = self.merge_cut(new_cut=extra_cut,**kwargs)
        else:
            cut = extra_cut
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
            plt.savefig(kwargs.get('output_folder')+f'modulation_phi{phi}{suf}.png')
        return phase, phase_err, modulation, modulation_err, chi2, cut

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
            
    def do_run(self, **kwargs):
        """
        """
        
        if kwargs.get('cut_type')=='rectangular':
            external_cut = self.get_cut(**kwargs)
            print(f'\nExternal cut = {external_cut}\n')
            coord_ = self.cut_string_rect(external_cut,expression='TRK_BAR')
            print(coord_)
            cut_suf = f'_cut_rect'
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape='rectangular', cut_edges=coord_, suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=coord_[0], suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=coord_[1], suf=cut_suf, **kwargs)
            
        if kwargs.get('cut_type')=='circle':
            external_cut = self.get_cut(**kwargs)
            print(f'\nExternal cut = {external_cut}\n')
            x,y,r = self.cut_string_circ(external_cut,expression='TRK_BAR')
            coord_ = [x,y,r]
            cut_suf = f'_cut_circ'
            print(coord_)
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape='circle', cut_edges=coord_, suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=[coord_[0]-coord_[2],coord_[0]+coord_[2]], suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=[coord_[1]-coord_[2],coord_[1]+coord_[2]], suf=cut_suf, **kwargs)
            
        if kwargs.get('cut_type')==None:
            cut_suf = ''
            self.map(expression='TRK_BAR', merge_cut=False,  extra_cut=cut_base, map_title='barycenter map', shape=None, cut_edges=None, suf=cut_suf, **kwargs)
            self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=None, suf=cut_suf, **kwargs)
            self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj', merge_cut=False, extra_cut=cut_base, cut_edges=None, suf=cut_suf, **kwargs)
        
        ecut, gauss_model = self.pha_spectrum_plot(figure_name='pha_spectrum', energy_cut=True, merge_cut = True, extra_cut = cut_base, suf=cut_suf, **kwargs) 

        track_size_cut='(TRK_SIZE > 0)'
        cut2= cut_logical_and(ecut,track_size_cut)


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

        print ("\ncut_final = ",cut_final,'\n')
        
        ###################################
        self.map(expression='TRK_BAR', merge_cut=True,  extra_cut=cut_final, map_title='barycenter map cut', shape=None, cut_edges=None, suf='_cut'+cut_suf, **kwargs)
        self.projections(coord='X', expression='TRK_BAR', figure_name='x bar proj cut', merge_cut=True, extra_cut=cut_final, cut_edges=None, suf='_cut'+cut_suf, **kwargs)
        self.projections(coord='Y', expression='TRK_BAR', figure_name='y bar proj cut', merge_cut=True, extra_cut=cut_final, cut_edges=None, suf='_cut'+cut_suf, **kwargs)
        
        
        ###################################
        # istogramma ph1
        phase_1, phase_err_1, modulation_1, modulation_err_1, chi2_1, cut_1 = self.modulation(phi=1, modulation_title='modulation phi 1', merge_cut=True, extra_cut=cut_final, suf=cut_suf, **kwargs)
        # istogramma ph2
        phase_2, phase_err_2, modulation_2, modulation_err_2, chi2_2, cut_2 = self.modulation(phi=2, modulation_title='modulation phi 2', merge_cut=True, extra_cut=cut_final, suf=cut_suf, **kwargs)

        self.pha_spectrum_plot(figure_name='pha_spectrum_cut', energy_cut=False, merge_cut = True, extra_cut = cut_final, suf=cut_suf, **kwargs)
        
        peak = gauss_model.parameter_value('Peak')
        peak_err = gauss_model.parameter_error('Peak')
        resolution = gauss_model.resolution()
        resolution_err = gauss_model.resolution_error()

        N_EVENTS_cut = self.run_list.num_events(cut_final)
        RUN_ID_ =  self.run_list.measurements_id()
        RUN_ID = RUN_ID_.split(' - ')[0]
        
        results = [peak, peak_err, resolution, resolution_err, phase_1, phase_err_1, modulation_1, modulation_err_1, chi2_1, phase_2, phase_err_2, modulation_2, modulation_err_2, chi2_2, cut_final,  kwargs.get('cut_type'), N_EVENTS_cut, RUN_ID]

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
