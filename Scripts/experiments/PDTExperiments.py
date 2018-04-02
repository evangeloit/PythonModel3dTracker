import os

import PythonModelTracker.PFSettings as pfs

import PythonModelTracker.PFHelpers.PFTrackingHelper as pfh

os.chdir(os.environ['bmbv']+"/Scripts/")


#Viz Parameters
visualize = {'enable':False,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize['client'] in ['opencv','blender']

# Model Params.
model_name = "human_ext"
meta_mult = 1.15  # use to change model scale if supported.
model3d,model_class = pfh.PFTracking.get_model(model_name)

# PF Params
reps = 5
n_particles = [100, 200, 400, 600,  800, 1000, 1200, 1400,1600]
enable_metaopt = [False]
pf_listener_flag = False

# Objective Params.
weighted_part_mult = 1            # Set to 1 for kinect objective, any other value for weighted objective HMF.
objective_weights = {'rendering':1.,
                     'primitives':0.,
                     'collisions':0.}
depth_cutoff = {'Object':50,'Human':500, 'Hand':150}
bgfg_type = {'Object':'depth',
             'Hand':'depth',
             'Human':'depth'}

# Dataset selection.
dids = {'Object':["marker_01"],
        'Hand': ["captured_free_motion1"],#["experiment_all_fing1"],#,"seq0","seq1"qui,"sensor"],
        'Human':["M1D1","M1D2","M1D4","F1D1","F1D2","F1D3","F1D4","F2D1","F2D2","F2D3","F2D4"]
                 #"mhad_s09_a08", "mhad_s09_a09", "mhad_s09_a10", "mhad_s09_a11","pdt_m1d1"]
        }


# Results Filename
res_filename = os.path.join(Paths.results, '{0}_tracking/pdt/test___{1}-{2}_p{3}_meta{4}_r{5}.json')

# Dataset Loop.
for em in enable_metaopt:
    for r in range(reps):
        for did in dids[model_class]:
            for np in n_particles:
                override_settings = {'n_particles': np}
                pf_params, meta_params = pfs.GetSettings(model3d, model_class,override_settings)

                tracker = pfh.PFTracking(model3d, model_class)
                tracker.load_dataset(did)
                tracker.init_pf(pf_params,meta_mult,pf_listener_flag)
                tracker.init_objective(weighted_part_mult,bgfg_type[model_class],objective_weights,depth_cutoff[model_class])
                if em: tracker.init_metaoptimizer(meta_params)
                tracker.loop(visualize)
                tracker.save_trajectory(res_filename.format(model_class, did, model3d.model_name,np,em,r))





