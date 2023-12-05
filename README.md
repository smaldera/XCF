# XCF
this repository should contail the software for the XCF project 
## 1 ANALISI SPETTRI
This section shows how to deal with the codes contained in **analisi_spetttri**.
### 1.1 plot_allSpectra.py
This code allows to plot together different type of spectra, for example spectra acquired using the CMOS ASI camera and the Silicon Drift Detector (SDD). The file format for these spectra is different and the main properties will be described in the following paragraphs.

This code needs a positional argument that represent the file.txt that is used by the code to get the spectra and every additional instruction. This file.txt is suggested to be a personal file since it will point towards the personal paths in the computer. To use plot_allSpectra.py you need to run

      python plot_allSpectra.py /path/to/your/file/file.txt

inside the analisi_spetttri folder.
#### 1.1.1 SDD
An example of a correct file.txt for the case of sdd spectra is the following:

    BASE_PATH=/path/to/your/folder/that/contains/the/data/

    FILE=File_name_1.mca sdd  
    ACQ_TIME  
    LEGEND=Name_of_this_spectrum  
    NORM_PEAK=norm_lower_limit norm_upper_limit  
    ADD_PLOT  
    FIT=fit_lower_limit fit_upper_limit gaussian_fit_amplitude gaussina_fit_peak gaussian_fit_sigma  

    FILE=File_name_2.mca sdd  
    ACQ_TIME  
    LEGEND=Name_of_this_spectrum  
    NORM_PEAK=norm_lower_limit norm_upper_limit  
    ADD_PLOT  
    FIT=fit_lower_limit fit_upper_limit gaussian_fit_amplitude gaussina_fit_peak gaussian_fit_sigma  

    STABILITY  

each numerical value has to be separated by the other by a single space, while no space is required between = and the following parameter. 
1) **BASE_PATH**: the path to your folder that contains the data. If you are going to plot spectra from different folders, remember to put here the most general path and then to specify the paths in the FILE=.
2) **FILE**: the path to the file that contains the spectrum. The spectrum file in the case of the SDD is **.mca**. Immediately after the path, separated from it by a single space, you have to specify the type of spectrum, so you have to type **sdd**.
3) **ACQ_TIME**: in the case of the SDD, the code will automatically search in the .mca file the acquisition time and the final spectrum will be a **normalized spectrum**.
4) **LEGEND**: simply the name of the spectrum that will be shown in the final plot. No spcae are allowed, only _ . If you comment this part putting *#* in front of LEGEND, the spectrum name will not be shown in the final plot.
5) **NORM_PEAK**: when you give or uncomment this instruction, the spectrum will be **normalized at the peak**. This instruction has to be followed by the _norm_lower_limit_ and _norm_upper_limit_, which tell to the code where it has to search the peak. These values have to be specified in _keV_.
6) **ADD_PLOT**: this instruction is the most important and it tells to the code to plot the spectrum.
7) **FIT**: when you give or uncomment this instruction, the peak of the spectrum will be fitted with a gaussian. In particular you have to specify:
   1) the energy [keV] range where you are searching the peak: _fit_lower_limit_ and _fit_lower_limit_, not necessary the same for the peak normalization.
   2) the normalization of the gaussian: _gaussian_fit_amplitude_.
   3) the energy [keV] of the peak of the gaussian: _gaussina_fit_peak_.
   4) the sigma [keV] of the peak of the gaussina: _gaussian_fit_sigma_.
8) **STABILITY**: this instruction has to be given as the last instruction of the file.txt





