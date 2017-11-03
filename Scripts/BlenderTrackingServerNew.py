import PythonModelTracker.PFHelpers.PFSettings as pfs
import PythonModelTracker.PFHelpers.TrackingTools as tt

#Viz Parameters
visualize = {'enable':True,
             'client': 'blender','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize['client'] in ['opencv','blender']

# Model & Dataset
dataset = 'pdt_m1d1'
model_name = 'mh_body_male'
model3d, model_class = tt.ModelTools.GenModel(model_name)
params_ds = tt.DatasetTools.Load(dataset)
grabber = tt.DatasetTools.GenGrabber(params_ds)
grabber_ldm = tt.DatasetTools.GenLandmarksGrabber(params_ds, model3d)

res_filename = None #os.path.join(paths.results,"{0}_tracking/{1}_{2}_p{3}_smartpf_ransac0102.json")

# PF Initialization
hmf_arch_type = "2levels"
pf_params = pfs.Load(model_name, model_class,hmf_arch_type)
pf_params['pf']['n_particles'] = 128
pf_params['init_state'] = tt.DatasetTools.GenInitState(params_ds, model3d)
pf_params['meta_mult'] = 1
pf_params['pf_listener_flag'] = False
#pf_params['pf']['smart_pf'] = True
#pf_params['pf']['levmar_particles'] = 1
#pf_params['pf']['obs_filter_ratios'] = [0.05, 0.25]
pf,rng = tt.ParticleFilterTools.GenPF(pf_params,model3d)

# Objectives
objective_params = {
    'objective_weights':{'rendering':0.5,
                         'primitives':0.5,
                         'collisions':0.
                         },
    'depth_cutoff': 500,
    'bgfg_type': 'depth'
}
model3dobj,mesh_manager = tt.ObjectiveTools.GenObjective(pf, model3d, objective_params)

tt.TrackingLoopTools.loop(params_ds,model3d,grabber,grabber_ldm,mesh_manager,pf,pf_params,model3dobj,objective_params,
                        visualize,res_filename)