import os

import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs
import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt
import PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools as vt
import PythonModel3dTracker.Paths as Paths

#Viz Parameters
visualize_params = {'enable':True,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize_params['client'] in ['opencv','blender']

# Model & Datasets
dataset = 'mhad_s01_a04'
model_name = 'mh_body_male_custom_0950'
model3d, model_class = tt.ModelTools.GenModel(model_name)
params_ds = tt.DatasetTools.Load(dataset)
enable_openpose_grabber = True


res_filename = os.path.join(Paths.results,"Human_tracking/Levmar/{0}.json".format(dataset))

# PF Initialization
hmf_arch_type = "2levels"
pf_params = pfs.Load(model_name, model_class,hmf_arch_type)
pf_params['pf']['n_particles'] = 5
pf_params['pf']['init_state'] = tt.DatasetTools.GenInitState(params_ds, model3d)
pf_params['meta_mult'] = 1
pf_params['pf_listener_flag'] = False
pf_params['pf']['smart_pf'] = True
pf_params['pf']['smart_particles'] = 15
pf_params['pf']['smart_pf_model'] = "COCO"
pf_params['pf']['smart_pf_interpolate_bones'] = ["R.UArm", "R.LArm","R.ULeg", "R.LLeg","L.UArm", "L.LArm","L.ULeg", "L.LLeg"]
pf_params['pf']['smart_pf_interpolate_num'] = 3
pf_params['pf']['obs_filter_ratios'] = [0.25, 0.5]

# Objectives
objective_params = {
    'enable': True, #pf_params['pf']['n_particles'] > pf_params['pf']['levmar_particles'],
    'objective_weights':{'rendering':1.,
                         'primitives':0.,
                         'collisions':0.
                         },
    'depth_cutoff': 500,
    'bgfg_type': 'depth'
}
mesh_manager = tt.ObjectiveTools.GenMeshManager(model3d)
model3dobj, decoder, renderer = tt.ObjectiveTools.GenObjective(mesh_manager, model3d, objective_params)
visualizer = vt.Visualizer(model3d, mesh_manager, decoder, renderer)
decoder = visualizer.decoder
grabbers = tt.DatasetTools.GenGrabbers(params_ds, model3d, enable_openpose_grabber)
pf, rng = tt.ParticleFilterTools.GenPF(pf_params, model3d, decoder)

results = tt.TrackingLoopTools.loop(params_ds,model3d, grabbers, pf,
                          pf_params['pf'],model3dobj,objective_params,
                          visualizer, visualize_params)

if res_filename is not None: results.save(res_filename)