import os
import sys
import itertools

import PythonModel3dTracker.Paths as Paths



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



model_names = ["mh_body_male_custom_0950", "mh_body_male_custom_1050", "mh_body_male_custom_1100",
               "mh_body_male_custom_0950", "mh_body_male_custom_1050", "mh_body_male_custom",
               "mh_body_male_custom_0900", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
               "mh_body_male_custom_0950", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
               "mh_body_male_custom_0950", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
               "mh_body_male_custom_0950", "mh_body_male_custom_0950", "mh_body_male_custom_0950",
               "mh_body_male_custom_0950", "mh_body_male_custom",      "mh_body_male_custom",
               "mh_body_male_custom"]
datasets = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04',
            'mhad_s04_a04', 'mhad_s05_a04', 'mhad_s06_a04',
            'mhad_s07_a04', 'mhad_s08_a04', 'mhad_s09_a01',
            'mhad_s09_a02', 'mhad_s09_a03', 'mhad_s09_a04',
            'mhad_s09_a05', 'mhad_s09_a06', 'mhad_s09_a07',
            'mhad_s09_a08', 'mhad_s09_a09', 'mhad_s09_a10',
            'mhad_s09_a11', 'mhad_s10_a04', 'mhad_s11_a04',
            'mhad_s12_a04'
]
#model_names = ["mh_body_male_custom"]
#datasets = ["mhad_ammar"]

# Command line parameters.
sel_rep = int(sys.argv[1])
dry_run = int(sys.argv[2])

# Experiment Parameters.
dataset_model_pairs = [(d, m) for (d, m) in zip(datasets, model_names)]
ransac = [[0.0, 0.15],[0.0, 0.3],[0.0, 0.45],[0.0, 0.6],
          [0.15, 0.15],[0.15, 0.3],[0.15, 0.45],[0.15, 0.6],
          [0.3, 0.3],[0.3, 0.45],[0.3, 0.6]]
levmar_particles = [1, 10, 20, 50]
n_particles = [0]

# Experiments loop.
rep = 0
for (dataset, model_name), r, lp, p in itertools.product(dataset_model_pairs, ransac, levmar_particles, n_particles):
    if p == 0: p = lp
    if rep == sel_rep:
        # Results Filename
        res_filename = os.path.join(Paths.results, "Human_tracking/Levmar/{0}_{1}_p{2}_lp{3}_ransac{4}.json")
        res_filename = res_filename.format( dataset, model_name, p, lp, r.__str__())

        if dry_run:
            print '{0} -- d: {1}, m:{2}, p:{3}, lp:{4}, ransac:{5}, results:{6}'.\
                format(rep, dataset, model_name, p, lp, r, res_filename)

        else:

            import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs
            import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt
            import PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools as vt

            model3d, model_class = tt.ModelTools.GenModel(model_name)
            params_ds = tt.DatasetTools.Load(dataset)


            # PF Initialization
            hmf_arch_type = "2levels"
            pf_params = pfs.Load(model_name, model_class, hmf_arch_type)
            pf_params['pf']['n_particles'] = p
            pf_params['pf']['init_state'] = tt.DatasetTools.GenInitState(params_ds, model3d)
            pf_params['meta_mult'] = 1
            pf_params['pf_listener_flag'] = False
            pf_params['pf']['obs_filter_ratios'] = r
            pf_params['pf']['smart_pf'] = (lp > 0)
            pf_params['pf']['smart_particles'] = lp
            pf_params['pf']['smart_pf_model'] = "COCO"
            pf_params['pf']['smart_pf_interpolate_bones'] = ["R.UArm", "R.LArm", "R.ULeg", "R.LLeg", "L.UArm", "L.LArm",
                                                             "L.ULeg", "L.LLeg"]
            pf_params['pf']['smart_pf_interpolate_num'] = 3


            enable_openpose_grabber = (lp > 0)

            #Performing tracking
            mesh_manager = tt.ObjectiveTools.GenMeshManager(model3d)
            model3dobj, decoder, renderer = tt.ObjectiveTools.GenObjective(mesh_manager, model3d, objective_params)
            visualizer = vt.Visualizer(model3d, mesh_manager, decoder, renderer)
            decoder = visualizer.decoder
            grabbers = tt.DatasetTools.GenGrabbers(params_ds, model3d, enable_openpose_grabber)
            pf, rng = tt.ParticleFilterTools.GenPF(pf_params, model3d, decoder)

            results = tt.TrackingLoopTools.loop(params_ds, model3d, grabbers, pf,
                                      pf_params['pf'], model3dobj, objective_params,
                                      visualizer, visualize_params)

            #for f in tracker.results.states:
            #    print(f,model_name,tracker.results.states[f][model_name])

            results.save(res_filename)
    rep += 1





