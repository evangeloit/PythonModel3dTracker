import os

import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs
import PythonModel3dTracker.PythonModelTracker..PFHelpers.PFTrackingHelper as pfh


#Viz Parameters
visualize = {'enable':True,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize['client'] in ['opencv','blender']

# Model Params.
model_name = "mh_body_male_custom"
meta_mult = .8  # use to change model scale if supported.
model3d,model_class = pfh.PFTracking.get_model(model_name)

# PF Params
enable_metaopt = False
pf_listener_flag = False

# Objective Params.
weighted_part_mult = 1            # Set to 1 for kinect objective, any other value for weighted objective HMF.
objective_weights = {'rendering':1,
                     'primitives':0.,
                     'collisions':0.,
                     'openpose':False}
depth_cutoff = {'Object':50,'Human':500, 'Hand':150}
bgfg_type = {'Object':'depth',
             'Hand':'depth',
             'Human':'depth'}

# Dataset selection.
did = {'Object':"box_01",
       'Hand': "seq0",
       'Human':"mhad_s02_a04"
}


# Results Filename
res_filename = None #os.path.join(paths.results,"box_01.json")

# Loading pf settings.
hmf_arch_type = "2levels"
pf_params = pfs.Load(model_name, model_class,hmf_arch_type)
pf_params['pf']['n_particles'] = 128

pf_params['pf']['smart_pf'] = False
pf_params['pf']['levmar_particles'] = 1
pf_params['pf']['obs_filter_ratios'] = [0.05, 0.25]

#Performing tracking
tracker = pfh.PFTracking( model3d, model_class)
tracker.load_dataset(did[model_class])
tracker.init_pf(pf_params,meta_mult,pf_listener_flag)
tracker.init_objective(weighted_part_mult,bgfg_type[model_class],objective_weights,depth_cutoff[model_class])
if enable_metaopt: tracker.init_metaoptimizer(pf_params)
tracker.loop(visualize)
#for f in tracker.results.states:
#    print(f,model_name,tracker.results.states[f][model_name])
if res_filename:
    tracker.save_trajectory(res_filename)





