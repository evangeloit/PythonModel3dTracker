import os
import sys

import PythonModelTracker.PFHelpers.PFSettings as pfs
import PythonModelTracker.PFHelpers.PFTrackingHelper as pfh

os.chdir(os.environ['bmbv']+"/Scripts/")


#Viz Parameters
visualize = {'enable':False,
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
objective_weights = {'rendering':0.9999,
                     'primitives':0.0001,
                     'collisions':0.,
                     'openpose':True}
depth_cutoff = {'Object':50,'Human':500, 'Hand':150}
bgfg_type = {'Object':'depth',
             'Hand':'depth',
             'Human':'depth'}

# Dataset selection.
did = {'Object':"Captured_calibration_box",
       'Hand': "seq0",
       'Human':"mhad_s02_a04"
}


ransac = [[0.0, 0.0]]
levmar_particles = [1]
n_particles = [2,3]
sel_rep = int(sys.argv[1])
rep = 0
for r in ransac:
    for lp in levmar_particles:
        for p in n_particles:
            if p == 0: p = lp
            if rep == sel_rep:
                # Results Filename
                res_filename = os.path.join(Paths.results, "{0}_tracking/Levmar/{1}_{2}_p{3}_lp{4}_ransac{5}.json")

                # Loading pf settings.
                hmf_arch_type = "2levels"
                pf_params = pfs.Load(model_name, model_class,hmf_arch_type)
                pf_params['pf']['n_particles'] = p
                pf_params['pf']['smart_pf'] = True
                pf_params['pf']['levmar_particles'] = lp
                pf_params['pf']['obs_filter_ratios'] = r

                #Performing tracking
                tracker = pfh.PFTracking( model3d, model_class)
                tracker.load_dataset(did[model_class])
                tracker.init_pf(pf_params,meta_mult,pf_listener_flag)
                tracker.init_objective(weighted_part_mult,bgfg_type[model_class],objective_weights,depth_cutoff[model_class])
                if enable_metaopt: tracker.init_metaoptimizer(pf_params)
                tracker.loop(visualize)
                #for f in tracker.results.states:
                #    print(f,model_name,tracker.results.states[f][model_name])
                res_filename = res_filename.format(model_class, did[model_class],
                                                   model3d.model_name, p, lp,r.__str__())
                if res_filename:
                    tracker.save_trajectory(res_filename)
            rep += 1





