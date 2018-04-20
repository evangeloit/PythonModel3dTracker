import PyMBVCore as core
import PyMBVParticleFilter as pf

#PFSubstate Initialization
def CreatePFSubState(rng,model3d, partition_name, n_particles, like_variance, std_dev):
    pft = pf.ParticleFilterSubState()
    pft.resampling_flag = True
    pft.like_variance = like_variance
    pft.n_particles = n_particles
    pft.state_est_method = pf.PFEstMethod.pf_est_max
    pft.initFromModel3d(model3d,str(partition_name))
    pft.rng = rng
    pft.std_dev = core.DoubleVector(std_dev)
    return pft

#AuxModel Initialization
def CreateAuxModel(rng,model3d, partition_name, part_name, model_params):
    pft = CreatePFSubState(rng, model3d, partition_name, model_params['n_particles'], model_params['like_variance'], model_params['std_dev'])
    aux = pf.AuxModel()
    aux.pf = pft
    aux.partition_name = str(partition_name)
    aux.part_name = part_name

    pm_dyn = pf.PropagationModelGaussian()
    pm_dyn.rng = rng
    pm_dyn.affected_dims = model3d.partitions.partitions[aux.partition_name]
    pm_dyn.std_dev = core.DoubleVector(model_params['std_dev'])
    aux.propagation_model_dynamic = pm_dyn
    if len(model_params['std_dev_cond'][aux.partition_name]) > 0:
        #aux.std_dev_cond = core.DoubleVector(model_params['std_dev_cond'])
        pm_cnd = pf.PropagationModelGaussian()
        pm_cnd.rng = rng
        pm_cnd.affected_dims = model3d.partitions.partitions[aux.partition_name]
        pm_cnd.std_dev = core.DoubleVector(model_params['std_dev_cond'][aux.partition_name])
        aux.propagation_model_conditional = pm_cnd
    return aux

#AuxModels Initializationn
def CreateAuxModelMap(rng,model3d,hmf_params):
    aux_models_map = {}


    # Creating partition_names list.
    partition_names = []
    for partition_name, child_partition_names in hmf_params['children_map'].items():
        if partition_name not in partition_names: partition_names.append(partition_name)
        for c in child_partition_names:
            if c not in partition_names: partition_names.append(c)

    # Creating the auxiliary models for the partitions.
    for partition_name in partition_names:
        part_name = hmf_params['model_parts_map'][partition_name]
        aux_models_map[partition_name] = CreateAuxModel(rng, model3d, partition_name, part_name, hmf_params)

    # Uses hmf_params.children_map to set the children of each aux_model.
    for partition_name,children_list in hmf_params['children_map'].items():
        for c in children_list:
            aux_models_map[partition_name].appendChild(aux_models_map[c])
    return aux_models_map


# class PFStdParams:
#     def __init__(self,n_particles,like_variance,std_dev):
#         self.n_particles = n_particles
#         self.like_variance = like_variance
#         self.std_dev = std_dev
#
#
# class AuxModelParams:
#     def __init__(self,n_particles,like_variance,std_dev,std_dev_cond):
#         self.n_particles = n_particles
#         self.like_variance = like_variance
#         self.std_dev = std_dev
#         self.std_dev_cond = std_dev_cond
#
#
# class MetaOptimizerParams:
#     def __init__(self,n_generations,n_particles,std_dev,affected_parts):
#         self.n_generations = n_generations
#         self.n_particles = n_particles
#         self.std_dev = std_dev
#         self.affected_parts = affected_parts
#
#
# class MetaFitterParams:
#     def __init__(self,max_hist_frames,max_hist_meta,n_skip_frames):
#         self.max_hist_frames = max_hist_frames
#         self.max_hist_meta = max_hist_meta
#         self.n_skip_frames = n_skip_frames
#
#
# class MetaParams:
#     def __init__(self,metaopt_params,metafit_params):
#         self.metaopt_params = metaopt_params
#         self.metafit_params = metafit_params
#
#
# class PFHMFParams:
#     def __init__(self,model_params_map, children_map, model_parts_map):
#         self.model_params_map = model_params_map
#         self.children_map = children_map
#         self.model_parts_map = model_parts_map


def CreatePF(rng,model3d,pf_params):
    if pf_params['type'] == "pf_std":
        return CreatePFStd(rng, model3d, pf_params)
    else:
        return CreatePFHMF(rng, model3d, pf_params)



def CreatePFHMF(rng,model3d,hmf_params):
    aux_models_map = CreateAuxModelMap(rng,model3d,hmf_params)
    hmf = pf.ParticleFilterHMF()
    hmf.aux_models = aux_models_map['main']
    return hmf


def CreatePFStd(rng,model3d,pfs_params):
    pfs = pf.ParticleFilterStandard()
    pfs.rng = rng
    pfs.resampling_flag = True
    pfs.like_variance = pfs_params['like_variance']
    pfs.n_particles = pfs_params['n_particles']
    pfs.state_est_method = pf.PFEstMethod.pf_est_max
    pfs.initFromModel3d(model3d)
    pfs.state = core.DoubleVector(pfs_params['init_state'])
    pfs.std_dev = core.DoubleVector(pfs_params['std_dev'])
    return pfs


def CreateMetaOptimizer(model3d,partition_name,mo_params):
    mo = pf.PFParticlesOptimizer()
    mo.initFromModel3d(model3d,partition_name)
    mo.n_generations = mo_params['n_generations']
    mo.n_particles = mo_params['n_particles']
    mo.std_dev = core.DoubleVector([mo_params['std_dev']]*sum(mo.active_dims))
    mo.affected_parts = core.StringVector(mo_params['affected_parts'])
    return mo


def CreateMetaFitter(model3d,partition_name,mf_params):
    mf = pf.ModelMetaFitting()
    mf.active_dims = model3d.partitions.partitions[partition_name]
    mf.max_hist_frames = mf_params['max_hist_frames']
    mf.max_hist_meta = mf_params['max_hist_meta']
    mf.n_skip_frames = mf_params['n_skip_frames']
    return mf
