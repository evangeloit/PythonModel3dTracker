import cv2
import numpy as np
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.Features2DUtils as f2d
import BlenderMBV.BlenderMBVLib.AngleTransformations as at
import PyModel3dTracker as pm3d
import copy


class ObjectAppearance:
    def __init__(self, dimensions = None, p3d_model = None, p3d_defpose = None, p2d_img = None, descriptors = None):
        self.dimensions = dimensions
        self.p3d_model = p3d_model
        self.p3d_defpose = p3d_defpose
        self.p2d_img = p2d_img
        self.descriptors = descriptors

class ObjectData:
    def __init__(self, model3d, object_appearance):
        self.model3d = model3d
        self.appearance = object_appearance
        self.default_state = model3d.default_state
        # Setting Scale
        self.default_state[7] = self.appearance.dimensions[0]
        self.default_state[8] = self.appearance.dimensions[1]
        self.default_state[9] = self.appearance.dimensions[2]


class RigidObjectDetectorORB:
    def __init__(self, objects_data, min_matches = 10, outliers_ratio = 0.5):

        self.objects_data = objects_data
        self.n_objects = len(objects_data)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.outliers_ratio = outliers_ratio
        self.states = []
        self.transformations = []
        self.inliers = []
        self.outliers = []


    def filter_depth(self, keypoints, descriptors, camera, depth):
        p3d_np, p2d_np = f2d.GetPointsFromKeypoints(keypoints, camera, depth)
        kp_mask = [p[2] > 0 for p in p3d_np.T]
        #print "kp_mask:,", kp_mask
        kp_filt = [kp for kp, m in zip(keypoints, kp_mask) if m]
        des_filt = descriptors[np.array(kp_mask)]
        return kp_filt, des_filt, p3d_np


    def detect(self,imgs,clbs):
        depth = imgs[0]
        img = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
        if len(imgs)==3: mask = imgs[2]
        else: mask = np.ones_like(img)
        camera = clbs[0]

        kp, des = self.orb.detectAndCompute(img, mask)
        kp, des, p3d = self.filter_depth(kp, des, clbs[0], depth)
        #print des.shape
        #p3d_def_vec = mbv.Core.Vector3fStorage(self.appearance.p3d_def.T)

        # BF Matching
        self.states = []
        self.transformations = []
        self.inliers = []
        self.outliers = []
        for oi in self.objects_data:
            matches = self.bf.match(des, oi.appearance.descriptors)
            # FLANN matching
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            # search_params = dict(checks = 50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matches = flann.match(des1,des2)
            matches = sorted(matches, key=lambda x: x.distance)
            matches_good = matches
            src_match_indices = [m.trainIdx for m in matches_good]
            dst_match_indices = [m.queryIdx for m in matches_good]

            if len(matches_good) > self.min_matches:
                src_pts_vec = mbv.Core.Vector3fStorage(
                    [mbv.Core.Vector3([oi.appearance.p3d_defpose[0][m.trainIdx],
                                       oi.appearance.p3d_defpose[1][m.trainIdx],
                                       oi.appearance.p3d_defpose[2][m.trainIdx]])
                     for m in matches_good
                    ]
                )
                dst_pts_vec = mbv.Core.Vector2fStorage(
                    [mbv.Core.Vector2([kp[m.queryIdx].pt[0],
                                       kp[m.queryIdx].pt[1]])
                     for m in matches_good]
                )
                src_pts_np = src_pts_vec.data.T
                dst_pts_np = p3d[:, dst_match_indices]
                Rt = RigidObjectDetectorORB.CalcRigidTransformRansac(src_pts_np, dst_pts_np)


                Trb = mbv.Core.DoubleVector()
                outliers_idx = mbv.Core.IntVector()
                pm3d.posest(Trb, dst_pts_vec, src_pts_vec, self.outliers_ratio, camera, outliers_idx)
                outliers_idx = [i for i in outliers_idx]

                state_cur = RigidObjectDetectorORB.calc_state_Rt(oi.default_state, Rt)
                #state_cur = RigidObjectDetectorORB.calc_state(oi.default_state, Trb)
                self.states.append(state_cur)
                self.transformations.append(Trb)
                cur_inliers = [p for i,p in enumerate(dst_pts_vec) if i not in outliers_idx ]
                self.inliers.append(cur_inliers)
                cur_outliers = [p for i,p in enumerate(dst_pts_vec) if i in outliers_idx ]
                self.outliers.append(cur_outliers)

            else:
                self.states.append(None)
                self.transformations.append(None)
                self.inliers.append([])
                self.outliers.append([])
        return self.states

    @staticmethod
    def calc_state_Rt(default_state, Rt):
        state = copy.deepcopy(default_state)

        _, _, angles, tr, _ = at.decompose_matrix(Rt)
        quat = at.quaternion_from_euler(angles[0], angles[1], angles[2])
        state[0] = tr[0]
        state[1] = tr[1]
        state[2] = tr[2]
        state[3] = quat[1]
        state[4] = quat[2]
        state[5] = quat[3]
        state[6] = quat[0]
        # print 'State:', state

        return state

    @staticmethod
    def calc_state(default_state,Trb):
        state = copy.deepcopy(default_state)
        R, _ = cv2.Rodrigues(np.array([Trb[0], Trb[1], Trb[2]]))

        # Rmat = np.eye(4,4,dtype=R.dtype)
        # Rmat[0:3][0:3] = R
        quat = at.quaternion_from_matrix(R)
        print 'Rmat:\n', R
        print 't:', Trb[3], Trb[4], Trb[5]
        state[0] = Trb[3]
        state[1] = Trb[4]
        state[2] = Trb[5]
        state[3] = quat[1]
        state[4] = quat[2]
        state[5] = quat[3]
        state[6] = quat[0]
        #print 'State:', state

        return state


    # GET Rigid Transform p->q.
    @staticmethod
    def CalcRigidTransformRansac(p, q, N = 5000, dmax = 30, M = 3):
        Np = np.shape(p)[1]
        inliers_ratio = 0
        for i in range(N):
            sel_points_idx = np.random.choice(Np, M)
            #print "sel_points_idx",sel_points_idx
            Rt = RigidObjectDetectorORB.CalcRigidTransformSVD(p[:, sel_points_idx], q[:, sel_points_idx])
            cur_inliers_idx, cur_ratio = RigidObjectDetectorORB.CalcInliersRatio(Rt, p, q, dmax)
            if cur_ratio >= inliers_ratio:
                inliers_ratio = cur_ratio
                inliers_idx = cur_inliers_idx
        #print "inliers_idx", inliers_idx
        Rt = RigidObjectDetectorORB.CalcRigidTransformSVD(p[:, inliers_idx],  q[:, inliers_idx])
        print "Ransac Rt:\n", Rt
        return Rt

    @staticmethod
    def CalcInliersRatio(Rt, p, q, dmax):
        n = np.shape(p)[1]
        qhom = np.append(q, np.ones((1,n)),axis=0)
        q_hom = np.dot(Rt, qhom)
        q_ = q_hom[0:3,:]
        q_[0,:] /= q_hom[3,:]
        q_[1,:] /= q_hom[3,:]
        q_[2,:] /= q_hom[3,:]
        eucl_dist = np.linalg.norm(p-q_, axis=0)
        #print "eucl_dist:", eucl_dist
        cur_inliers_idx = np.where(eucl_dist <= dmax)[0]
        ratio = np.count_nonzero(eucl_dist <= dmax) / float(n)
        return cur_inliers_idx, ratio

    # GET Rigid Transform p->q.
    @staticmethod
    def CalcRigidTransformSVD(p, q):
        d = np.shape(q)[0]
        n = np.shape(q)[1]
        qm = (np.mean(q, axis=1)[np.newaxis]).T
        pm = (np.mean(p, axis=1)[np.newaxis]).T
        Y = q - np.tile(qm, (1, n))
        X = p - np.tile(pm, (1, n))
        S = np.dot(X, Y.T)
        [U, _, V] = np.linalg.svd(S)
        VU = np.dot(V, U.T)
        detVU = np.linalg.det(VU)
        I = np.eye(d)
        I[d-1, d-1] = detVU
        R = np.dot(np.dot(V,I), U.T)
        t = qm - np.dot(R, pm)
        Rt = at.compose_matrix(angles=at.euler_from_matrix(R), translate=t.flatten())
        #print 'p:\n', p
        #print 'q:\n', q
        #print 'SVD Rt:\n', Rt
        return Rt




