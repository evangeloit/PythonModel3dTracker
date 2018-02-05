import numpy as np
import PythonModel3dTracker.PyMBVAll as mbv
import PyCeresIK as IK
from PythonModel3dTracker.PythonModelTracker.OpenPoseGrabber import OpenPoseGrabber
import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFInitialization as pfi
import copy


class SmartPF:

    default_smart_particles = 10
    default_filter_ratios = [0.1, 0.2]

    def __init__(self,rng,model3d,pf_params,decoder=None,landmarks=None,model2keypoints=None):

        self.model3d = model3d
        #self.std_dev = pf_params['std_dev']
        self.pf = SmartPF.CreatePF(rng,model3d,pf_params)#SmartPF.CreatePF(rng,model3d,pf_params)
        #self.pf.dynamic_model = mpf.DynamicModel(self.DynamicModel)

        if (decoder is not None and landmarks is not None and model2keypoints is not None):
            self.decoder = decoder
            self.ba = SmartPF.CreateBA(model3d,decoder,landmarks,model2keypoints)
        else:
            self.decoder = None
            self.ba = None

        self.lnames = None
        self.landmarks = None
        self.keypoints2d = None
        self.keypoints3d = None
        self.calib = None
        self.particles_prev = None
        self.model3dobj = None
        self.smart_particles = SmartPF.default_smart_particles
        self.filter_ratios = SmartPF.default_filter_ratios
        if 'smart_particles' in pf_params:
            self.smart_particles = min(pf_params['smart_particles'], pf_params['n_particles'])
        if 'obs_filter_ratios' in pf_params:
            self.filter_ratios = pf_params['obs_filter_ratios']



    def __del__(self):
        mbv.Core.CachedAllocatorStorage.clear()


    def track(self,state,objective):
        self.pf.particles = mbv.PF.ParticlesMatrix(self.DynamicModel(self.pf.particles.particles))
        self.pf.track(state, objective)

    @staticmethod
    def CreateBA(model3d,decoder,landmarks):
        ba = IK.ModelAwareBundleAdjuster()
        lmv = IK.LandmarksVector()
        for l in landmarks:
            lmv.append(l)
        ba.decoder = decoder
        ba.low_bounds = model3d.low_bounds
        ba.high_bounds = model3d.high_bounds
        # assert (type(decoder) is mbv.Dec.GenericDecoderGPU) or \
        #        (type(decoder) is mbv.Dec.StateTransformDecoder)
        # if type(decoder) is mbv.Dec.GenericDecoderGPU:
        #     ba.decoder = decoder
        #     ba.low_bounds = model3d.low_bounds
        #     ba.high_bounds = model3d.high_bounds
        # else:
        #     ba.decoder = decoder.delegate
        #     st = decoder.state_transformer
        #     ba.low_bounds = st.process(model3d.low_bounds, mbv.Dec.StateTransformDirection.Forward)
        #     ba.high_bounds = st.process(model3d.high_bounds, mbv.Dec.StateTransformDirection.Forward)
        #     print 'highBounds length:', len(ba.high_bounds)
        ba.landmarks = lmv
        ba.ceres_report = False
        ba.max_iterations = 500
        # ba.soft_bounds = True

        return ba

    @staticmethod
    def CreatePF(rng,model3d,pf_params):
        pfs = mbv.PF.ParticleFilterStandard()
        pfs.rng = rng
        pfs.resampling_flag = True
        pfs.like_variance = pf_params['like_variance']
        pfs.n_particles = pf_params['n_particles']
        pfs.state_est_method = mbv.PF.PFEstMethod.pf_est_max
        pfs.initFromModel3d(model3d)
        pfs.state = mbv.Core.DoubleVector(pf_params['init_state'])
        pfs.std_dev = mbv.Core.DoubleVector(pf_params['std_dev'])
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
            #print 'SmartPF Dynamic: LEVMAR'
            for i in range(self.smart_particles):
                keypoints_cur = OpenPoseGrabber.FilterKeypointsRandom(self.keypoints3d,
                                                                      self.keypoints2d,
                                                                      self.filter_ratios)
                #print 'keypoints_cur:',keypoints_cur
                observations = OpenPoseGrabber.ConvertIK([keypoints_cur], self.calib)
                # do something with
                cur_state = mbv.Core.DoubleVector(particles[:, i])

                #print 'cur_state:', cur_state
                score, results = self.ba.solve(observations[0], cur_state)

                particles[:,i] = results
        #for f in range(n_particles):
        #    particles[0, f] += 100*f
        #print 'particles after:', particles[0:3,:]
        return particles


    def Objective(self,param_vectors):
        n_particles = len(param_vectors)
        n_dims = len(param_vectors[0])

        res = mbv.Core.DoubleVector([0]*n_particles)
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