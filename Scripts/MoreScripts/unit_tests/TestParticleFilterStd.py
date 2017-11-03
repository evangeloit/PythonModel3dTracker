import PyMBVCore as core
import PyMBVOptimization as opt
import PyMBVParticleFilter as pf


def dummy_obj(param_vectors):
    n_particles = len(param_vectors)
    n_dims = len(param_vectors[0])
    assert n_dims == 27
    res_vector = core.DoubleVector([0.1]*n_particles)
    res_vector[2] = 0.3
    res_vector[0] = 0.5
    return res_vector

obj = opt.ParallelObjective(dummy_obj)


model3d = pf.Model3dMeta.create(Paths.models + '/hand_std/hand_std.xml')



pfs = pf.ParticleFilterStandard()
pfs.n_particles = 10
pfs.n_dims = 4
pfs.like_variance = 0.01
solution = core.DoubleVector([1, 2, 3, 4])
pfs.state = solution
pfs.state_est_method = pf.PFEstMethod.pf_est_avg
pfs.calcStateEst()
print pfs.state
pfs.state_est_method = pf.PFEstMethod.pf_est_max
pfs.calcStateEst()
print pfs.state

print pfs.particles.particles

