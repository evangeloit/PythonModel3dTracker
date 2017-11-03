import csv
import numpy as np
import pylab as pl
import PFFileIO as pfio
from collections import namedtuple

res_id = '010'
saved_fig_filename_templ = './media/pf/results/meta_res_%s/figures/res_%s_%s.pdf' 
exp_info_xml = './media/pf/results/meta_res_%s/params/experiments_params.xml' % res_id
results_filename='./media/pf/results/meta_eval_%s.txt' % res_id
#sel_generation_num=[ [2],[2], [2], [20] ]
#sel_method=['std','prt','hmf','pso']
method_cola=['r', 'g', 'b','m','y','c','k','#FF0010','#00FF10']
method_colb=['r', 'g', 'b','m','y','c','k']
expl_label= ['']
metric_label=['E(mm)','C','C2','Es']
linewidths= [3, 3, 3, 3, 3, 3, 3, 3, 3]
#sel_generation_num=[ [1],[1],[1],[30] ]
#sel_method=['pf-hmf', 'pf-prt', 'pf-std', 'pso-std']

SelData = namedtuple("SelData", "param_names values")
sel_data = [SelData(['mop','mft','mhm'],['1','1','20'])]
            

save_fig = True
legend_on = True
legend_loc=1 #0:bottom right 1:upper right, 2:upper_left
fontsize=14

plot_against='mhf'
if plot_against == 'thr':
    xlabel='thres'
    xlim=[0,1]
    xticks = np.arange(0,1,0.1)
    xbounds=[0,100]
if plot_against == 'mul':
    xlabel='MetaMult'
    xlim=[0,2.25]
    xticks = np.arange(0,2.25,0.25)
    xbounds=[0,100]
if plot_against == 'mhm':
    xlabel='Ns'
    xlim=[1,12]
    xticks = np.arange(0,12,1)
    xbounds=[0,100]
if plot_against == 'mhf':
    xlabel='Nf'
    xlim=[-1,100]
    xticks = np.arange(0,105,25)
    xbounds=[0,100]
if plot_against == 'lnr':
    xlabel='Noise Ratio'
    xlim=[0,0.6]
    xticks = np.arange(0,0.6,0.1)
    xbounds=[0,100]
    
    
sel_metric=3 #0:Error 3:Meta norm   
ylabel=metric_label[sel_metric]
ylim_metric=[[20,42],[0,0],[0,0],[0.5,2.5]]
ystep=[5,0,0,0.5]
ylim = ylim_metric[sel_metric]
yticks = np.arange(ylim[0],ylim[1]+ystep[sel_metric],ystep[sel_metric])

