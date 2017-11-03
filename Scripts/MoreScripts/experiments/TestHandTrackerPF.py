#!/usr/bin/python
import sys
import subprocess
import PFFileIO as pfio
import itertools as it
import os
import shutil

def ask_yn(ask_string):
    user_ans = 'a'
    while ((user_ans != 'y') and (user_ans != 'n')):
        user_ans = raw_input(ask_string)
    return (user_ans == 'y')

param_loc = './params/'
param_files = ['experiments_params.xml', 
               'HandTrackerPF.ini', 
               'pf_config.xml']

ei = pfio.load_experiments_info(param_loc+param_files[0])

params_folder = ei.out_folder+'params/'


if not os.path.exists(params_folder):
    create_folder = ask_yn('Create folder <%s>? ' % params_folder)
    if (create_folder):
        print 'Creating folder <%s>' % params_folder
        os.makedirs(params_folder)
    else:
        sys.exit()
else:
    print 'Folder <%s> already exists.' % params_folder    
    
copy_params = ask_yn('Copy params to <%s>?' % params_folder)
if copy_params:        
    for f in param_files:
        print 'Copying param file <%s> to <%s>.' % (f,params_folder)
        shutil.copyfile(param_loc+f,params_folder+f)

start_exps = ask_yn('Start experiments?')
    
#Calulating all the param combinations using the cross product.
options_cp = list()
for s in ei.exp_params:
    exp_params_values = list()
    exp_params_names = list()
    for i in s:
        exp_params_values.append(i.values)
        exp_params_names.append(i.name)
    options_cp = options_cp + list(it.product(*exp_params_values))
print 'options_cp: ',options_cp

opt_idx = exp_params_names.index('opt')
type_idx = exp_params_names.index('pf_type')
g_idx = exp_params_names.index('n_generations')
p_idx = exp_params_names.index('n_particles')


exp_counter=0
for param_tuple in options_cp:
    param = list(param_tuple)
    t = param[type_idx]
    g = int(param[g_idx])
    p_init = int(param[p_idx])
    p = int(p_init / g)
    if (t=='prt'):
        p = p / ei.prt_part_num
    if (t=='hmf'):
        p = p / ei.hmf_part_num
    param[g_idx] = g
    param[p_idx] = p
    fname_token = '_rep%04d'
    args_token = ' -v 0'
    for idx,i in enumerate(exp_params_names):
        #print idx, i
        fname_token_cur = ei.exp_params[0][idx].fname_token % param_tuple[idx]
        fname_token = fname_token + fname_token_cur
        args_token_cur = ' ' + ei.exp_params[0][idx].cmd + ' ' + str(param[idx])
        args_token = args_token + args_token_cur
    #print fname_token
    #print args_token
    
    for r in range(ei.start_rep,ei.stop_rep+1):
        exp_counter = exp_counter + 1
        filename = ei.out_folder + ei.out_filename
        cur_fname_token = fname_token % r
        filename = filename % (cur_fname_token);        
        cur_args_token = args_token + ' -o ' + filename        
        if (os.path.isfile(filename)):
            print 'Skipping <%s>, file already exists.' % (filename)
        else:
            full_command = ei.command + cur_args_token
            print '%04d, %s' % (exp_counter, full_command)
            if (start_exps>0):
                subprocess.call(full_command, shell=True)            
