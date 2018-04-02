import PythonModelTracker.PFSettings as pfs

import PythonModelTracker.PFHelpers.PFTrackingHelper as pfh

#Viz Parameters
visualize = {'enable':False,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize['client'] in ['opencv','blender']

# Model Params.
model_name = "hand_skinned_rds"
meta_mult = 1.  # use to change model scale if supported.
model3d,model_class = pfh.PFTracking.get_model(model_name)

# PF Params
enable_metaopt = False
pf_listener_flag = False

# Objective Params.
weighted_part_mult = 1            # Set to 1 for kinect objective, any other value for weighted objective HMF.
objective_weights = {
    'hmfpen05':{'rendering':0.95,'primitives':0.05,'collisions':0.},
    'hmfpen10':{'rendering':0.90,'primitives':0.10,'collisions':0.},
    'hmfpen20':{'rendering':0.80,'primitives':0.20,'collisions':0.},
    'hmfpen30':{'rendering':0.70,'primitives':0.30,'collisions':0.},
    'hmfpen40':{'rendering':0.60,'primitives':0.40,'collisions':0.},
    'hmfpen50':{'rendering':0.50,'primitives':0.50,'collisions':0.},
    'hmfpen60':{'rendering':0.40,'primitives':0.60,'collisions':0.},
    'hmfpen70':{'rendering':0.30,'primitives':0.70,'collisions':0.},
    'hmfpen80':{'rendering':0.20,'primitives':0.80,'collisions':0.},
    'hmfpen90':{'rendering':0.10,'primitives':0.90,'collisions':0.},
    'hmfpen100':{'rendering':0.00,'primitives':1.00,'collisions':0.}
}
depth_cutoff = 150
bgfg_type = 'depth'

# Dataset selection.
datasets_xml = Paths.ds_info + "/ht_datasets.xml"
dids = ["experiment_two_fing1","experiment_all_fing1","experiment_index_middle"]
n_particles = [5,10,20,40]
methods = ['hmfplain','hmfpen']

# Results Filename
res_filename = Paths.results + 'Hand_tracking/rds/' + \
               'method_{0}_pf_particles_{1}_dataset_{2}_pf_type_hmf_{3}.json'

# Dataset Loop.
for did in dids:
    for r in range(5):
        for m in objective_weights:
            for np in n_particles:
                print(did,m,np)
                override_settings = {'n_particles': np}
                pf_params, meta_params = pfs.GetSettings(model3d, model_class,override_settings)

                tracker = pfh.PFTracking(model3d, model_class)
                tracker.load_dataset(datasets_xml,did)
                tracker.init_pf(pf_params,meta_mult,pf_listener_flag)
                tracker.init_objective(weighted_part_mult,bgfg_type,objective_weights[m],depth_cutoff)
                if enable_metaopt: tracker.init_metaoptimizer(meta_params)
                tracker.loop(visualize)
                tracker.save_trajectory(res_filename.format(m,np,did,r))





