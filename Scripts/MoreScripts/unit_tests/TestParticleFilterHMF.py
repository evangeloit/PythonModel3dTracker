import os

import PyMBVCore as core
import PyMBVOptimization as opt
import PyMBVParticleFilter as pf
import PythonModelTracker.PFSettings as pfs

import PythonModelTracker.PFHelpers.PFInitialization as pfi

os.chdir(os.path.join(os.environ['bmbv'],'Scripts'))

def dummy_obj(param_vectors):
    n_particles = len(param_vectors)
    n_dims = len(param_vectors[0])
    assert n_dims == 27
    res_vector = core.DoubleVector([0.1]*n_dims)
    res_vector[2] = 0.3
    res_vector[0] = 0.5
    return res_vector

solution = core.DoubleVector()
obj = opt.ParallelObjective(dummy_obj)
model3d = pf.Model3dMeta.create(Paths.model3d_dict['hand_skinned'][1])
model_class  = Paths.model3d_dict['hand_skinned'][0]

rng = pf.RandomNumberGeneratorOpencv()

pf_params, meta_params = pfs.GetSettings(model3d, model_class)
#aux_models_map = pfi.CreateAuxModelMap(rng, model3d,pf_params)

hmf = pfi.CreatePFHMF(rng, model3d,pf_params)
hmf.track(solution,obj)


