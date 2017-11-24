import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs
import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt

#Viz Parameters
visualize = {'enable':True,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize['client'] in ['opencv','blender']

# Model & Dataset
dataset = 'mhad_s01_a04'
model_name = 'mh_body_male_custom'
model3d, model_class = tt.ModelTools.GenModel(model_name)
params_ds = tt.DatasetTools.Load(dataset)
openpose_grabber = True


res_filename = None #os.path.join(paths.results,"{0}_tracking/{1}_{2}_p{3}_smartpf_ransac0102.json")

# PF Initialization
hmf_arch_type = "2levels"
pf_params = pfs.Load(model_name, model_class,hmf_arch_type)
pf_params['pf']['n_particles'] = 10
pf_params['pf']['init_state'] = tt.DatasetTools.GenInitState(params_ds, model3d)
pf_params['meta_mult'] = 1
pf_params['pf_listener_flag'] = False
pf_params['pf']['smart_pf'] = True
pf_params['pf']['levmar_particles'] = 10
pf_params['pf']['obs_filter_ratios'] = [0.05, 0.25]
# if smart_pf: pf,rng = tt.ParticleFilterTools.GenSmartPF(pf_params, model3d)
# else: pf,rng = tt.ParticleFilterTools.GenPF(pf_params,model3d)

# Objectives
objective_params = {
    'objective_weights':{'rendering':0.5,
                         'primitives':0.,
                         'collisions':0.
                         },
    'depth_cutoff': 500,
    'bgfg_type': 'depth'
}
mesh_manager = tt.ObjectiveTools.GenMeshManager(model3d)
model3dobj = tt.ObjectiveTools.GenObjective(mesh_manager, model3d, objective_params)
grabbers = tt.DatasetTools.GenGrabbers(params_ds, model3d, openpose_grabber)
pf,rng = tt.ParticleFilterTools.GenPF(pf_params, model3d, model3dobj.decoder)


tt.TrackingLoopTools.loop(params_ds,model3d, grabbers, mesh_manager,pf,
                          pf_params['pf'],model3dobj,objective_params,
                        visualize,res_filename)