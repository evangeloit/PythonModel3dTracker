import cv2
import numpy as np
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv
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




    def detect(self,imgs,clbs):
        depth = imgs[0]
        img = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
        if len(imgs)==3: mask = imgs[2]
        else: mask = np.ones_like(img)
        camera = clbs[0]

        kp, des = self.orb.detectAndCompute(img, mask)
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

                Trb = mbv.Core.DoubleVector()
                outliers_idx = mbv.Core.IntVector()
                pm3d.posest(Trb, dst_pts_vec, src_pts_vec, self.outliers_ratio, camera, outliers_idx)
                outliers_idx = [i for i in outliers_idx]
                # print('RigidBody:', Trb)
                #print('Outliers Indices:', outliers_idx)
                self.states.append(RigidObjectDetectorORB.calc_state(oi.default_state, Trb))
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
    def calc_state(default_state,Trb):
        state = copy.deepcopy(default_state)
        R, _ = cv2.Rodrigues(np.array([Trb[0], Trb[1], Trb[2]]))

        # Rmat = np.eye(4,4,dtype=R.dtype)
        # Rmat[0:3][0:3] = R
        quat = at.quaternion_from_matrix(R)
        #print 'Rmat', R
        state[0] = Trb[3]
        state[1] = Trb[4]
        state[2] = Trb[5]
        state[3] = quat[1]
        state[4] = quat[2]
        state[5] = quat[3]
        state[6] = quat[0]
        #print 'State:', state

        return state


