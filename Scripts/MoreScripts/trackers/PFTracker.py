import PythonModelTracker.PFSettings as pfs
import PythonModelTracker.PFHelpers.PFTrackingHelper as pfh
import os
os.chdir(os.environ['hts'])

#Viz Parameters
visualize = {'enable':True, 'labels':False, 'depth':True, 'rgb':True}
wait_time = 0                       # cv::waitkey

# Model Params.
model_name = "mh_body_male"
meta_mult = 0.9  # use to change model scale if supported.
model3d,model_class = pfh.PFTracking.get_model(model_name)

# PF Params
pf_params, meta_params = pfs.GetSettings(model3d, model_class)
enable_metaopt = False
pf_listener_flag = False

# Objective Params.
weighted_part_mult = 1            # Set to 1 for kinect objective, any other value for weighted objective HMF.
objective_weights = {'rendering':1.,
                     'primitives':0.,
                     'collisions':0.0}
depth_cutoff = {'Human':600, 'Hand':150}
bgfg_type = {'Hand':'depth',
             'Human':'depth'}

# Dataset selection.
datasets_xml = {"Hand":"ds_info/ht_datasets.xml",
                "Human":"ds_info/bt_datasets.xml"}
dids = {'Hand': ["iasonas"],#,"seq0","seq1","sensor"],
        'Human':["kostas_good_01"]#,"mhad_s09_a02","mhad_s09_a03","mhad_s09_a04","mhad_s09_a05","mhad_s09_a06","mhad_s09_a07",
                 #"mhad_s09_a08", "mhad_s09_a09", "mhad_s09_a10", "mhad_s09_a11"]
        }

# Results Filename
res_filename = ['','rs/{0}_tracking/{1}.txt'][1]

# Dataset Loop.
for did in dids[model_class]:
    tracker = pfh.PFTracking(model3d, model_class)
    tracker.load_dataset(datasets_xml[model_class],did)
    tracker.init_pf(pf_params,meta_mult,pf_listener_flag)
    tracker.init_objective(weighted_part_mult,bgfg_type[model_class],objective_weights,depth_cutoff[model_class])
    if enable_metaopt: tracker.init_metaoptimizer(meta_params)
    tracker.loop(visualize,wait_time)
    tracker.save_trajectory(res_filename.format(model_class, did))




