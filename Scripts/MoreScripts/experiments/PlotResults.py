import csv
import numpy as np
import pylab as pl
import PFFileIO as pfio
from collections import namedtuple

res_id = '017'
saved_fig_filename = './media/pf/results/res_%s/figures/res_C10_budget.pdf' % res_id
exp_info_xml = './media/pf/results/res_%s/params/experiments_params.xml' % res_id
results_filename='./media/pf/results/eval_%s_C10.txt' % res_id
#sel_generation_num=[ [2],[2], [2], [20] ]
#sel_method=['std','prt','hmf','pso']
method_cola=['g', 'r', 'm','b','y','c','k']
method_colb=['g', 'r', 'm','b','y','c','k']
expl_label= ['PSO','HMF','PFS','PRT']
metric_label=['E(mm)','C','Ec(mm)']
linewidths= [3, 3, 3, 3]
#sel_generation_num=[ [1],[1],[1],[30] ]
#sel_method=['pf-hmf', 'pf-prt', 'pf-std', 'pso-std']

SelData = namedtuple("SelData", "param_names values")
sel_data = [SelData(['opt','typ','gen','lnr','dns'],['pso','std','0028','0','0']),
            SelData(['opt','typ','gen','lnr','dns'],['pf','hmf','0001','0','0']),
            SelData(['opt','typ','gen','lnr','dns'],['pf','std','0001','0','0']),
            SelData(['opt','typ','gen','lnr','dns'],['pf','prt','0001','0','0'])]
            #SelData(['opt','typ','gen','dns','lnr'],['pf','prt','0001','0','0'])]

legend_on = False
legend_loc=2 #0:bottom right 1:upper right, 2:upper_left
fontsize=14


plot_against='prt'
if plot_against == 'prt':
    xlabel='budget'
    xlim=[350,2450]
    xticks = np.arange(400,2450,200)
    xbounds=[1,12]
    
if plot_against == 'lnr':
    xlabel='Noise Ratio'
    xlim=[0,0.65]
    xticks = np.arange(0,0.65,0.1)
    xbounds=[0,24]

    
sel_metric=1 #0:Error 1:Succ ratio, 2:error succ.    
ylabel=metric_label[sel_metric]
ylim_metric=[[5,14],[0.2,1.0]]
ystep=[1,0.1]
ylim = ylim_metric[sel_metric]
yticks = np.arange(ylim[0],ylim[1]+ystep[sel_metric],ystep[sel_metric])



#Fps for PSO,HMF,PFS,PRT 400:200:1600
plot_against_fps = False
if plot_against_fps:
    fps = [[-1,49, 43,38,35,32,29,26,25,24,22,21],
           [-1,98, 76,60,44,40,38,33,28,27,23,22],
           [-1,132,85,62,45,36,30,26,22,19,17,14],    
           [-1,105,78,62,46,44,35,34,30,25,23,22]]
    #fps = [[40,30,20],
    #       [60,50,40,30,20]]
    xlim=[20,110]
    xticks = np.arange(20,120,10)
    xlabel = 'fps'
    
ei = pfio.load_experiments_info(exp_info_xml)

results = []
with open(results_filename) as inputfile:
    results = list(csv.reader(inputfile))

metric = list();
for elem in results:
     metric.append(elem[sel_metric+1])
     #succ_rate.append(elem[2])
     #mse_succ.append(elem[3])
#print mse

#convert mse to np array
metric_f=list()
for el in metric:
    metric_f.append(float(el))
metric_np = np.array(metric_f)
print 'metric_np:',metric_np

filenames=list();
for elem in results:
    filenames.append(elem[0])

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
    
#print params_names
#print params_values

x_idx = params_names.index(plot_against)
#print x_idx

#mth_np = np.array(mth)
x_vals = np.array(params_values[x_idx])
#print 'x_vals:', x_vals 
#print mth_np

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : fontsize}

pl.rc('font', **font)
fig1 = pl.figure()
ax1 = fig1.add_subplot(111)


for m_idx,sel_d in enumerate(sel_data):
    #print sel_d
    
    #Applying filters to data.
    sel_ind = np.array([True]*x_vals.size)
    for f_idx,sel_f in enumerate(sel_d.param_names):
        sel_v = sel_d.values[f_idx]
        f_pidx = params_names.index(sel_f)
        #print 'sel_f: ', sel_f, 'f_pidx: ', f_pidx, 'sel_v: ',sel_v
        for v_idx,cur_v in enumerate(params_values[f_pidx]):
            sel_ind[v_idx] = (sel_ind[v_idx] & (sel_v == cur_v))  
    #print 'sel_ind:',sel_ind
    
    if (sum(sel_ind) > 0):
        sel_metric_np = metric_np[sel_ind]
        sel_x_vals = x_vals[sel_ind]
        #print 'sel_metric_np:',sel_metric_np
        #print 'sel_x_vals:',sel_x_vals
            
        mean_mse_np = np.array([])
        stddev_mse_np = np.array([])
        mean_x_vals = np.array([])
        for pn_val in np.unique(sel_x_vals):
            test_ind = (sel_x_vals == pn_val)
            mean_mse = np.mean(sel_metric_np[test_ind])
            stddev_mse = np.std(sel_metric_np[test_ind])
            #print mean_mse
            mean_mse_np = np.append(mean_mse_np,mean_mse)
            stddev_mse_np = np.append(stddev_mse_np,stddev_mse)
            mean_x_vals = np.append(mean_x_vals,pn_val)    

        #print mean_mse_np
        #pl.plot(mean_x_vals,mean_mse_np,'go')
        #pl.show()
        #ax1.plot(sel_x_vals,sel_metric_np,method_colb[m_idx]+'o')
        #ax1.plot(mean_x_vals,mean_mse_np,method_cola[m_idx], 
        #         label=sel_d.values[0]+'-'+sel_d.values[1])
        #ax1.plot(mean_x_vals,mean_mse_np-stddev_mse_np,method_cola[m_idx]+'x', 
        #         label=sel_d.values[0]+'-'+sel_d.values[1])
        #ax1.plot(mean_x_vals,mean_mse_np+stddev_mse_np,method_cola[m_idx]+'x', 
        #         label=sel_d.values[0]+'-'+sel_d.values[1])
        if (plot_against_fps):
            mean_x_vals = fps[m_idx]
            print 'mean_x_vals:',mean_x_vals
        print 'mean_x_vals:',mean_x_vals
        ax1.errorbar(mean_x_vals[xbounds[0]:xbounds[1]],
                     mean_mse_np[xbounds[0]:xbounds[1]],
                     yerr=stddev_mse_np[xbounds[0]:xbounds[1]],
                     color=method_cola[m_idx],
                     label=expl_label[m_idx],
                     linewidth=linewidths[m_idx])#sel_d.values[0]+'-'+sel_d.values[1]+'-'+sel_d.values[2])

ax1.set_ylim(ylim)
ax1.set_xlim(xlim)
pl.xticks(xticks)
pl.yticks(yticks)
#ax1.grid()
pl.gca().yaxis.grid(True)
if legend_on: 
    pl.legend(loc=legend_loc)
pl.xlabel(xlabel)
pl.ylabel(ylabel)
pl.show()
fig1.savefig(saved_fig_filename, bbox_inches='tight')
    
    

