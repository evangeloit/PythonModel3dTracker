import csv
import numpy as np
import pylab as pl
import PFFileIO as pfio
from collections import namedtuple

res_id = '005'
saved_fig_filename = './media/pf/results/meta_res_%s/figures/res_seq_error.pdf' % res_id
exp_info_xml = './media/pf/results/meta_res_%s/params/experiments_params.xml' % res_id
results_filename='./media/pf/results/metaseq_eval_005.txt'

#sel_generation_num=[ [2],[2], [2], [20] ]
#sel_method=['std','prt','hmf','pso']
method_cola=['g', 'b', 'r','m','y','c','k']
method_colb=['g', 'b', 'r','m','y','c','k']
expl_label= ['HMF','SOP','SFT']
plot_title=['Pose Error', 'Shape Error']
linewidths= [1, 1, 1, 1]
results_indices=[[2,6,10],[3,7,11]]

save_fig = True
legend_on = True
legend_loc=1 #0:bottom right 1:upper right, 2:upper_left
fontsize=14


   
ylabel=['E(mm)', 'Es']
ystep=5
ylim = [[0,40],[0,2]]
y_clip = 38


ei = pfio.load_experiments_info(exp_info_xml)


metric = []
with open(results_filename) as inputfile:
    metric = list(csv.reader(filter(lambda row: row[0]!='#', inputfile)))
print 'Found, <', len(metric), '> error sequences.'
    
    
max_len = 0
for elem in metric:
    max_len = max(max_len,len(elem))
print 'max_length: ', max_len
n_frames = max_len
x_vals = np.arange(n_frames) 

metric_np = np.array([])
for elem in metric:
    cur_metric=np.array(elem,dtype='float')
    cur_frames=cur_metric.shape[0]
    metric_np = np.append(metric_np,cur_metric)
    
metric_np = np.reshape(metric_np, [len(metric),n_frames])
print metric_np.shape
print x_vals.shape
        
        


xlabel='frame number'
xlim=[0,n_frames+10]
xticks = np.arange(0,n_frames,50)

    
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : fontsize}

pl.rc('font', **font)
fig1 = pl.figure()


n_plots = len(results_indices)
for plot_idx in range(n_plots):
    print plot_idx
    ax = pl.subplot(n_plots,1,plot_idx+1)
    
    for m_idx,r_idx in enumerate(results_indices[plot_idx]):
        pl.plot(x_vals,metric_np[r_idx,:],
             color=method_cola[m_idx],
             label=expl_label[m_idx],
             linewidth=linewidths[m_idx])#sel_d.values[0]+'-'+sel_d.values[1]+'-'+sel_d.values[2])
    
        ax.set_ylim(ylim[plot_idx])
        ax.set_xlim(xlim)
        pl.xticks(xticks)
        ax.grid(True)
        ax.set_title(plot_title[plot_idx])
        if (plot_idx == 0): 
            ax.legend(loc=legend_loc)
        if(plot_idx == n_plots-1):
            pl.xlabel(xlabel)
        else:
            ax.xaxis.set_ticklabels([])
        pl.ylabel(ylabel[plot_idx])
pl.show()
if save_fig:
    print 'Saving Fig to <', saved_fig_filename, '>.'
    fig1.savefig(saved_fig_filename, bbox_inches='tight')
    
    

