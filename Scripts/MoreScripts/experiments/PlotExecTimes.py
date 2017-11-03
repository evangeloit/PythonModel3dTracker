"""Plot the execution times (fps) loaded from a csv file.
csv format: n_generations,n_particles,fps
budget (x-values) should be provided.
"""

import pylab as pl
import numpy as np
import csv

#Setting input file.
csv_path ='rs/exec_times_new.txt'

xlim=[350,1650]

#Initializing
x = np.array([200,400,600,800,1000,1200,1400,1600])
gen=np.empty(0,int)
prt=np.empty(0,int)
fps=np.empty(0,float)

#Reading csv
#with open(csv_path, "rb") as f_obj:
#    csv_reader = csv.reader(f_obj)    
#    for idx,row in enumerate(csv_reader):
#        if idx > 0:
#            gen = np.append(gen,int(row[0]))
#            prt = np.append(prt,int(row[1]))
#            fps = np.append(fps,float(row[2]))

#Plotting
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 14}

pl.rc('font', **font)
#fig1 = pl.figure(1)
#ax1 = fig1.add_subplot(111)
#for n_gen in np.unique(gen):
#    sel_ind = (gen == n_gen)
#    sel_fps = fps[sel_ind]
#    print sel_fps
#    ax1.plot(x,sel_fps,label='g'+str(n_gen))
#ax1.grid()
#pl.legend(loc=1)
#pl.xlabel('budget')
#pl.ylabel('fps')
#ax1.set_ylim([20,200])
#pl.show()

"""Old PlotExecTimes
"""
method_cola=['g', 'r', 'm','b','y','c','k']
x = np.array([400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400])
xlim=[350,2450]
ylim=[10,120]
yticks = np.arange(ylim[0],ylim[1]+10,10)
xticks = np.arange(400,2600,200)

pso_g28= [49.1207,45.9474,43.0219,39.847,38.4541,36.531,34.9663,33.2668,32.1574,30.446,28.9277,27.1918,26.4992,25.7,25,24.5,24,23,22,21.3,21]
pf_hmf = [97.5515,78.7002,75.5116,65.0999,60.0673,54.7345,43.6091,40.7,43.1816,35.877,37.8344,31.3765,33.4213,29.8,28,27.7,27,25,23,22.4,22]
pf_std = [131.822,104.877,84.7314,70.917,61.7322,52.203,45.3432,40.3812,36.3108,32.4971,30.248,27.6686,25.8846,24.4,22,20.5,19,18,17,16,14]    
pf_prt = [105.541,83.5998,77.7545,69.7253,62.5391,56.9346,45.9031,42.8614,44.3027,41.232,34.8918,34.5829,33.6564,32,30,27,25,24,23,22.3,22]

#Old values
#pso_g28= [41,36,32,28,26,23,21]
#pf_std = [102,69,49,38,30,25,22]    
#pf_prt = [79,60,49,36,33,28,27]
#pf_hmf = [78,59,47,34,32,29,26]
fig2 = pl.figure(2)
ax2 = fig2.add_subplot(111)

#ax1.plot(x,pso_g7,label='pso_gen7')
#ax1.plot(x,pso_g14,label='pso_gen14')
#ax1.plot(x,pso_g21,label='pso_gen21')
ax2.plot(x,pso_g28,label='PSO',color=method_cola[0],linewidth=3)
ax2.plot(x,pf_hmf,label='HMF',color=method_cola[1],linewidth=3)
ax2.plot(x,pf_std,label='PFS',color=method_cola[2],linewidth=3)
ax2.plot(x,pf_prt,label='PRT',color=method_cola[3],linewidth=3)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.grid()
pl.legend(loc=1)
#pl.xlabel('budget')
#pl.ylabel('fps')
pl.xticks(xticks)
pl.yticks(yticks)
#ax1.set_ylim([20,200])
pl.show()
fig2.savefig('cur_exec_times.pdf', bbox_inches='tight')
