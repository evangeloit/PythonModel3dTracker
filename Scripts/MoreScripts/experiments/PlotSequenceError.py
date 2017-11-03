import csv
import numpy as np
import pylab as pl
import PFFileIO as pfio
from collections import namedtuple

res_id = '018'
saved_fig_filename = './media/pf/results/res_%s/figures/res_seq_error.pdf' % res_id
exp_info_xml = './media/pf/results/res_%s/params/experiments_params.xml' % res_id
results_filename='./media/pf/results/eval_seq_%s.txt' % res_id
#sel_generation_num=[ [2],[2], [2], [20] ]
#sel_method=['std','prt','hmf','pso']
method_cola=['g', 'r', 'm','b','y','c','k']
method_colb=['g', 'r', 'm','b','y','c','k']
expl_label= ['PSO','HMF','PFS','PRT']
plot_title=['Low Noise', 'Mid Noise', 'High Noise']
linewidths= [1, 1, 1, 1]
results_indices=[[0,1],[2,3],[4,5]]

legend_on = True
legend_loc=1 #0:bottom right 1:upper right, 2:upper_left
fontsize=14


   
ylabel='E(mm)'
ystep=5
ylim = [0,40]
y_clip = 38
yticks = np.arange(ylim[0],ylim[1]+ystep,ystep)

ei = pfio.load_experiments_info(exp_info_xml)


metric = []
with open(results_filename) as inputfile:
    metric = list(csv.reader(filter(lambda row: row[0]!='#', inputfile)))    
    
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
    if (cur_frames == n_frames):
        cur_metric = np.clip(cur_metric,0,y_clip)
        metric_np = np.append(metric_np,cur_metric)
    else:    
        cur_x = (n_frames / cur_frames) * np.arange(cur_frames,dtype='float')
        #print 'type cur_x:', type(cur_x[0])
        #print 'type cur_metric:', type(cur_metric[0])
        cur_interp_metric = np.interp(x_vals,cur_x,cur_metric)
        cur_interp_metric = np.clip(cur_interp_metric,0,y_clip)
        metric_np = np.append(metric_np,cur_interp_metric)
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

n_plots = len(results_indices)
for plot_idx in range(n_plots):
    print plot_idx
    ax = pl.subplot(n_plots,1,plot_idx+1)
    
    for m_idx,r_idx in enumerate(results_indices[plot_idx]):
        pl.plot(x_vals,metric_np[r_idx,:],
             color=method_cola[m_idx],
             label=expl_label[m_idx],
             linewidth=linewidths[m_idx])#sel_d.values[0]+'-'+sel_d.values[1]+'-'+sel_d.values[2])
    
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        pl.xticks(xticks)
        pl.yticks(yticks)
        ax.grid(True)
        ax.set_title(plot_title[plot_idx])
        if (plot_idx == 0): 
            ax.legend(loc=legend_loc)
        if(plot_idx == n_plots-1):
            pl.xlabel(xlabel)
        else:
            ax.xaxis.set_ticklabels([])
        
        pl.ylabel(ylabel)
pl.show()
#fig1.savefig(saved_fig_filename, bbox_inches='tight')
    
    

