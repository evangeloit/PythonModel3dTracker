import numpy as np
import PythonModel3dTracker.PyMBVAll as mbv
import PyCeresIK as IK
from PythonModel3dTracker.PythonModelTracker.OpenPoseGrabber import OpenPoseGrabber
from PythonModel3dTracker.PythonModelTracker.LandmarksCorrespondences import model_landmark_partitions
import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFInitialization as pfi
import copy
import time





class SmartPF:
    model_partitions = {
        "mh_body_male_custom": ["global_pos", "r_arm", "l_arm", "r_leg", "l_leg", "head"],
        "mh_body_male_customquat": ["global_pos", "r_arm", "l_arm", "r_leg", "l_leg", "head"],
        "mh_body_male_custom_vector": ["global_pos", "r_arm", "l_arm", "r_leg", "l_leg", "head"]
    }

    default_smart_particles = 10
    default_filter_ratios = [0.1, 0.2]

    def __init__(self,rng,model3d,pf_params,decoder=None,landmarks=None):

        self.model3d = model3d
        #self.std_dev = pf_params['std_dev']
        self.pf = SmartPF.CreatePF(rng,model3d,pf_params)#SmartPF.CreatePF(rng,model3d,pf_params)
        #self.pf.dynamic_model = mpf.DynamicModel(self.DynamicModel)

        self.decoder = None
        self.ba = None
        if (decoder is not None and landmarks is not None):
            self.decoder = decoder
            self.ba = SmartPF.CreateBA(model3d,decoder,landmarks)

        self.lnames = None
        self.landmarks = None
        self.keypoints2d = None
        self.keypoints3d = None
        self.calib = None
        self.particles_prev = None
        self.model3dobj = None
        self.smart_particles = SmartPF.default_smart_particles
        self.filter_ratios = SmartPF.default_filter_ratios
        if 'smart_particles' in pf_params['smart_pf']:
            self.smart_particles = min(pf_params['smart_pf']['smart_particles'], pf_params['n_particles'])
        if 'obs_filter_ratios' in pf_params['smart_pf']:
            self.filter_ratios = pf_params['smart_pf']['obs_filter_ratios']



    def __del__(self):
        mbv.Core.CachedAllocatorStorage.clear()


    def track(self,state,objective):
        self.pf.particles = mbv.PF.ParticlesMatrix(self.DynamicModel(self.pf.particles.particles))
        self.pf.track(state, objective)
        return state


    def setLandmarks(self, landmark_names, landmarks):
        self.lnames = landmark_names
        self.landmarks = landmarks
        if self.ba is not None: self.ba.landmarks = landmarks


    @staticmethod
    def SetObservationBlocks(ba, model3d, obs_landmark_source, obs_landmark_names):

        mlp = model_landmark_partitions[(obs_landmark_source, model3d.model_name)]
        obs_blocks = IK.ObservationBlocks()
        for n in model3d.partitions.partition_names:
            cur_partition_mask = []
            for l in obs_landmark_names:
                if mlp[l] == n: cur_partition_mask.append(1)
                else: cur_partition_mask.append(0)
            obs_blocks.append(mbv.Core.UIntVector(cur_partition_mask))
            print n, cur_partition_mask
        ba.observation_blocks = obs_blocks

    @staticmethod
    def CreateBA(model3d,decoder,landmarks,
                 params={"enable_blocks" : False,
                         "ceres_report": False,
                         "max_iterations": 50}):

        if params["enable_blocks"]:
            ba = IK.ModelAwareBABlocks()
        else:
            ba = IK.ModelAwareBA()

        lmv = IK.LandmarksVector()
        for l in landmarks: lmv.append(l)
        ba.decoder = decoder

        partitions = mbv.PF.StatePartition()
        for n, p in zip(model3d.partitions.partitions.keys(), model3d.partitions.partitions.values()):
            if n in SmartPF.model_partitions[model3d.model_name]:
                partitions.addNamedPartition(n, model3d.partitions.partitions[n])
        model3d.partitions = partitions
        ba.model3d = model3d
        ba.low_bounds = model3d.low_bounds
        ba.high_bounds = model3d.high_bounds

        ba.landmarks = lmv
        ba.ceres_report = params["ceres_report"]
        ba.max_iterations = params["max_iterations"]
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
                t1 = time.time()
                score, results = self.ba.solve(observations[0], cur_state)
                print "Opt fps:", 1.0 / (time.time() - t1)

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
