import numpy as np
import PyMBVCore as Core
import PyMBVDecoding
import PyMBVRendering
import PyMBVAcquisition
import PyMBVParticleFilter as mpf
import PyCeresIK as IK
from PythonModelTracker.OpenPoseGrabber import OpenPoseGrabber
import PythonModelTracker.PFHelpers.PFInitialization as pfi
import copy


class SmartPF:

    def __init__(self,rng,model3d,pf_params,decoder=None,landmarks=None,model2keypoints=None):

        self.model3d = model3d
        #self.std_dev = pf_params['std_dev']
        self.pf = pfi.CreatePF(rng,model3d,pf_params)#SmartPF.CreatePF(rng,model3d,pf_params)
        #self.pf.dynamic_model = mpf.DynamicModel(self.DynamicModel)

        if (decoder is not None and landmarks is not None and model2keypoints is not None):
            self.ba = SmartPF.CreateBA(model3d,decoder,landmarks,model2keypoints)
        else:
            self.ba = None

        self.keypoints2d = None
        self.keypoints3d = None
        self.calib = None
        self.particles_prev = None
        self.model3dobj = None
        self.levmar_particles = 10
        self.filter_ratios = [0.1, 0.2]
        if 'levmar_particles' in pf_params:
            self.levmar_particles = pf_params['levmar_particles']
        if 'obs_filter_ratios' in pf_params:
            self.filter_ratios = pf_params['obs_filter_ratios']



    def __del__(self):
        Core.CachedAllocatorStorage.clear()

    @staticmethod
    def CreateBA(model3d,decoder,landmarks,model2keypoints):
        ba = IK.ModelAwareBundleAdjuster()
        lmv = IK.LandmarksVector()
        for l in landmarks:
            lmv.append(l)

        ba.decoder = decoder
        ba.landmarks = lmv
        ba.model_to_keypoints = model2keypoints
        ba.ceres_report = False
        ba.max_iterations = 500
        # ba.soft_bounds = True
        ba.low_bounds = model3d.low_bounds
        ba.high_bounds = model3d.high_bounds
        return ba

    @staticmethod
    def CreatePF(rng,model3d,pf_params):
        pfs = mpf.ParticleFilterStandard()
        pfs.rng = rng
        pfs.resampling_flag = True
        pfs.like_variance = pf_params['like_variance']
        pfs.n_particles = pf_params['n_particles']
        pfs.state_est_method = mpf.PFEstMethod.pf_est_max
        pfs.initFromModel3d(model3d)
        pfs.std_dev = Core.DoubleVector(pf_params['std_dev'])
        return pfs

    def DynamicModel(self, particles):
        n_particles = particles.shape[1]

        self.particles_prev = copy.deepcopy(particles)
        #results = Core.DoubleVector(n_particles)
        #print 'particles bef:', particles[0:3, :]
        #if self.std_dev is not None:
        #    print 'SmartPF Dynamic: perturbation, std_dev:', self.std_dev
        #    particles = np.random.normal(particles, np.matrix(self.std_dev).T)
            #for f in range(n_particles):
            #    particles[2, f] = 500*f
            #print 'particles before:', particles[0:3, :]


        if self.keypoints3d is not None:
            print 'SmartPF Dynamic: LEVMAR'
            for i in range(min(self.levmar_particles, n_particles)):
                keypoints_cur = OpenPoseGrabber.FilterKeypointsRandom(self.keypoints3d,
                                                                      self.keypoints2d,
                                                                      self.filter_ratios)
                #print 'keypoints_cur:',keypoints_cur
                observations = OpenPoseGrabber.ConvertIK([keypoints_cur], self.calib)
                # do something with
                cur_state = Core.DoubleVector(particles[:, i])
                #print 'cur_state:', cur_state
                score, results = self.ba.solve(observations[0], cur_state)
                #print 'levmar solve:', results
                particles[:,i] = results
        #for f in range(n_particles):
        #    particles[0, f] += 100*f
        #print 'particles after:', particles[0:3,:]
        return particles


    def Objective(self,param_vectors):
        n_particles = len(param_vectors)
        n_dims = len(param_vectors[0])

        res = Core.DoubleVector([0]*n_particles)
        # if self.model3dobj is not None:
        #     print 'SmartPF Objective running', 'n_particles:', n_particles, 'n_dims:', n_dims
        #     res = self.model3dobj.evaluate(param_vectors, 0)
        lbv = self.model3d.low_bounds
        hbv = self.model3d.high_bounds
        #
        lb = np.array(lbv)
        hb = np.array(hbv)
        for i, state in enumerate(param_vectors):
            st = np.array(state)
            st_prev = self.particles_prev[:,i]

            normalized_st = (st - lb) / (hb - lb )
            normalized_stprev = (st_prev - lb) / (hb - lb)
            normalized_dist = np.linalg.norm(normalized_st-normalized_stprev) / np.sqrt(n_dims)
            # print 'st:', st[0:3], 'st_prev:', st_prev[0:3]
            # print 'normalized dist:', normalized_dist
            res[i] = normalized_dist
        return res