import os
import sys
import itertools

import PythonModel3dTracker.Paths as Paths

landmarks_source = ['gt', 'detections', 'openpose', 'json_openpose'][3]

#Viz Parameters
visualize_params = {'enable':False,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize_params['client'] in ['opencv','blender']



# Objective Params.
objective_params = {
    'enable': True, #pf_params['pf']['n_particles'] > pf_params['pf']['levmar_particles'],
    'objective_weights':{'rendering':1.,
                         'primitives':0.,
                         'collisions':0.
                         },
    'depth_cutoff': 500,
    'bgfg_type': 'depth'
}




# model_names = ["mh_body_male_custom_0950", "mh_body_male_custom_1050", "mh_body_male_custom_1100",
#                "mh_body_male_custom_0950", "mh_body_male_custom_1050", "mh_body_male_custom",
#                "mh_body_male_custom_0900", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
#                "mh_body_male_custom_0950", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
#                "mh_body_male_custom_0950", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
#                "mh_body_male_custom_0950", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
#                "mh_body_male_custom_0950", "mh_body_male_custom",      "mh_body_male_custom",
#                "mh_body_male_custom"]
datasets = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04',
            'mhad_s04_a04', 'mhad_s05_a04', 'mhad_s06_a04',
            'mhad_s07_a04', 'mhad_s08_a04', 'mhad_s09_a01',
            'mhad_s09_a02', 'mhad_s09_a03', 'mhad_s09_a04',
            'mhad_s09_a05', 'mhad_s09_a06', 'mhad_s09_a07',
            'mhad_s09_a08', 'mhad_s09_a09', 'mhad_s09_a10',
            'mhad_s09_a11', 'mhad_s10_a04', 'mhad_s11_a04',
            'mhad_s12_a04'
]
model_names = ["mh_body_male_customquat"] * len(datasets)
#model_names = ["mh_body_male_custom"]
#datasets = ["mhad_ammar"]

# Command line parameters.
sel_rep = int(sys.argv[1])
dry_run = int(sys.argv[2])

# Experiment Parameters.
dataset_model_pairs = [(d, m) for (d, m) in zip(datasets, model_names)]
ransac = [  [0.0, 0.0] ]
#[[0.0, 0.15],[0.0, 0.3],[0.0, 0.45],[0.0, 0.6],
# [0.15, 0.15],[0.15, 0.3],[0.15, 0.45],[0.15, 0.6],
# [0.3, 0.3],[0.3, 0.45],[0.3, 0.6]]
levmar_particles = [1] #[1, 10, 20, 50]
n_particles = [0]
filter_occluded = [True, False]
filter_history = [True, False]

# Experiments loop.
rep = 0
for (dataset, model_name), r, lp, p, fo, fh in itertools.product(dataset_model_pairs, ransac, levmar_particles, n_particles, filter_occluded, filter_history):
    if p == 0: p = lp
    if rep == sel_rep:
        # Results Filename
        res_filename = os.path.join(Paths.results, "Human_tracking/Levmar/{0}_{1}_p{2}_lp{3}_ransac{4}_fo{5}_fh{6}.json")
        res_filename = res_filename.format( dataset, model_name, p, lp, r.__str__(), fo, fh)

        if dry_run:
            print '{0} -- d: {1}, m:{2}, p:{3}, lp:{4}, ransac:{5}, fo:{6}, fh:{7}, results:{8}'.\
                format(rep, dataset, model_name, p, lp, r, fo, fh, res_filename)

        else:
            print '{0} -- d: {1}, m:{2}, p:{3}, lp:{4}, ransac:{5}, fo:{6}, fh:{7}, results:{8}'. \
                format(rep, dataset, model_name, p, lp, r, fo, fh, res_filename)

            import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs
            import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt
            import PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools as vt

            model3d, model_class = tt.ModelTools.GenModel(model_name)
            params_ds = tt.DatasetTools.Load(dataset)
            params_ds.limits = [2, 5]


            # PF Initialization

            # PF Initialization
            hmf_arch_type = "2levels"
            pf_params = pfs.Load(model_name, model_class, hmf_arch_type)
            pf_params['pf']['n_particles'] = p
            pf_params['pf']['init_state'] = tt.DatasetTools.GenInitState(params_ds, model3d)
            pf_params['meta_mult'] = 1
            pf_params['pf_listener_flag'] = False
            pf_params['pf']['enable_smart'] = True
            pf_params['pf']['smart_pf']['smart_particles'] = lp
            pf_params['pf']['smart_pf']['enable_blocks'] = False
            pf_params['pf']['smart_pf']['enable_bounds'] = True
            pf_params['pf']['smart_pf']['ceres_report'] = False
            pf_params['pf']['smart_pf']['max_iterations'] = 50
            pf_params['pf']['smart_pf']['interpolate_num'] = 3
            pf_params['pf']['smart_pf']['filter_occluded'] = fo
            pf_params['pf']['smart_pf']['filter_occluded_params'] = {
                'thres': 0.2,
                'cutoff': 50,
                'sigma': 0.2
            }
            pf_params['pf']['smart_pf']['filter_random'] = False
            pf_params['pf']['smart_pf']['filter_history'] = fh
            pf_params['pf']['smart_pf']['filter_history_thres'] = 100




            #Performing tracking
            mesh_manager = tt.ObjectiveTools.GenMeshManager(model3d)
            model3dobj, decoder, renderer = tt.ObjectiveTools.GenObjective(mesh_manager, model3d, objective_params)
            visualizer = vt.Visualizer(model3d, mesh_manager, decoder, renderer)
            decoder = visualizer.decoder
            grabbers = tt.DatasetTools.GenGrabbers(params_ds, model3d, landmarks_source)
            pf, rng = tt.ParticleFilterTools.GenPF(pf_params, model3d, decoder)

            results = tt.TrackingLoopTools.loop(params_ds, model3d, grabbers, pf,
                                                pf_params['pf'], model3dobj, objective_params,
                                                visualizer, visualize_params)

            results.save(res_filename)

    rep += 1





