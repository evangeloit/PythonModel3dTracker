import csv
import numpy as np
import pylab as pl
import PFFileIO as pfio
from collections import namedtuple

import plot_meta_eval_011_all as pl_params



    
ei = pfio.load_experiments_info(pl_params.exp_info_xml)

results = []
with open(pl_params.results_filename) as inputfile:
    results = list(csv.reader(inputfile))

results[1]
metric = list();
for elem in results:
     metric.append(elem[pl_params.sel_metric+1])
     #succ_rate.append(elem[2])
     #mse_succ.append(elem[3])
#print mse

#convert metric to [1 x n_files] np array
metric_f=list()
for el in metric:
    metric_f.append(float(el))
metric_np = np.array(metric_f)
print 'Extracting metric <', pl_params.ylabel ,'>, metric size:',len(metric_np)

filenames=list();
for elem in results:
    filenames.append(elem[0])

#Creating Lists: param_names [1        x n_params]
#                param_values[n_params x n_files]
params_values=list()
params_names=list()
mth=list()
for fidx, filename in enumerate(filenames):
    dirs_filename = filename.split("/")
    filename_only = dirs_filename[len(dirs_filename)-1]
    #print filename_only
    tokens = filename_only.split("_")
    tokens.pop(0)
    for pidx,t in enumerate(tokens):
        if (fidx == 0):
            params_names.append(t[0:3])    
            params_values.append(list())
        #remove filename extension '.txt'
        if (pidx == len(tokens)-1):
            t = t[0:len(t)-5]
            
        params_values[pidx].append(t[3:len(tokens[pidx])])        
    
x_idx = params_names.index(pl_params.plot_against)
x_vals = np.array(params_values[x_idx])


font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : pl_params.fontsize}

pl.rc('font', **font)
fig1 = pl.figure()
ax1 = fig1.add_subplot(111)


for m_idx,sel_d in enumerate(pl_params.sel_data):
    #print sel_d
    
    #Applying filters to data.
    sel_ind = np.array([True]*x_vals.size)
    print 'Extracting selected data: ', sel_d.param_names , ' -> ', sel_d.values
    for f_idx,sel_f in enumerate(sel_d.param_names):
        sel_v = sel_d.values[f_idx]
        f_pidx = params_names.index(sel_f)
        for v_idx,cur_v in enumerate(params_values[f_pidx]):
            sel_ind[v_idx] = (sel_ind[v_idx] & ((sel_v == cur_v)|(sel_v == "all")))
        print 'sel feat:', sel_f, ' sel value:',sel_v, ' sel number:', sum(sel_ind)  
    print 'Total selected: ',sum(sel_ind), ' out of ', len(x_vals)
    
    if (sum(sel_ind) > 0):
        sel_metric_np = metric_np[sel_ind]
        sel_x_vals = x_vals[sel_ind]
        #print 'sel_metric_np:',sel_metric_np
        #print 'sel_x_vals:',sel_x_vals
            
        mean_mse_np = np.array([])
        stddev_mse_np = np.array([])
        mean_x_vals = np.array([])
        seldata_perpoint = []
        for pn_val in np.unique(sel_x_vals):
            test_ind = (sel_x_vals == pn_val)
            seldata_perpoint.append(sum(test_ind))
            mean_mse = np.mean(sel_metric_np[test_ind])
            stddev_mse = np.std(sel_metric_np[test_ind])
            #print mean_mse
            mean_mse_np = np.append(mean_mse_np,mean_mse)
            stddev_mse_np = np.append(stddev_mse_np,stddev_mse)
            mean_x_vals = np.append(mean_x_vals,pn_val)    

        
        print "mean_x_vals <{a:s}>:{b:s}".format(a=pl_params.plot_against,b=mean_x_vals)
        np.set_printoptions(precision=2)
        print 'mean_mse_np <', pl_params.ylabel ,'>:',mean_mse_np
        print 'Selected data per point:',seldata_perpoint
        ax1.errorbar(mean_x_vals[pl_params.xbounds[0]:pl_params.xbounds[1]],
                     mean_mse_np[pl_params.xbounds[0]:pl_params.xbounds[1]],
                     yerr=0,#stddev_mse_np[xbounds[0]:xbounds[1]],
                     color=pl_params.method_cola[m_idx],
                     label=pl_params.expl_label[m_idx],
                     linewidth=pl_params.linewidths[m_idx])#sel_d.values[0]+'-'+sel_d.values[1]+'-'+sel_d.values[2])

ax1.set_ylim(pl_params.ylim)
ax1.set_xlim(pl_params.xlim)
pl.xticks(pl_params.xticks)
pl.yticks(pl_params.yticks)
#ax1.grid()
pl.gca().yaxis.grid(True)
if pl_params.legend_on: 
    pl.legend(loc=pl_params.legend_loc)
pl.xlabel(pl_params.xlabel)
pl.ylabel(pl_params.ylabel)
pl.show()
if (pl_params.save_fig):
    saved_fig_filename = pl_params.saved_fig_filename_templ % (pl_params.res_id, pl_params.ylabel, pl_params.xlabel)
    print 'Saving Fig to <', saved_fig_filename, '>.'
    fig1.savefig(saved_fig_filename, bbox_inches='tight')
    
    

