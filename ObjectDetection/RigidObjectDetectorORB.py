import cv2
import numpy as np
np.set_printoptions(precision=1, suppress = True)
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.Features2DUtils as f2d
import PythonModel3dTracker.ObjectDetection.RigidObjectPosest3D as RigidObjectPosest3D
import PythonModel3dTracker.ObjectDetection.StateVectorTools as StateVectorTools
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
    def __init__(self, objects_data=None, settings=None):#min_matches = 10, inliers_ratio = 0.5):
        assert settings is not None
        self.settings = settings
        self.check_settings()
        if objects_data is not None:
            self.objects_data = objects_data
            self.n_objects = len(objects_data)

        if self.settings["features_type"] == 'orb':
            self.detector = cv2.ORB_create()
            self.detector.setMaxFeatures(self.settings["max_features"])
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        elif self.settings["features_type"] == 'sift':
            self.detector = cv2.xfeatures2d.SIFT_create(nfeatures=self.settings["max_features"])
            self.bf = cv2.BFMatcher()



        #self.min_matches = min_matches
        #self.inliers_ratio = inliers_ratio
        self.states = []
        #self.transformations = []
        self.inliers = []
        self.outliers = []

    def check_settings(self):
        assert "max_features" in self.settings
        assert "method" in self.settings
        assert "min_matches" in self.settings
        assert "inliers_ratio" in self.settings
        assert "features_type" in self.settings
        assert self.settings["method"] in ["2d3d", "3d3d"]
        assert self.settings["features_type"] in ["sift", "orb"]




    def filter_depth(self, keypoints, descriptors, camera, depth):
        if len(keypoints) > 0:
            p3d_np, p2d_np = f2d.GetPointsFromKeypoints(keypoints, camera, depth)
            kp_mask = np.array([p[2] > 0 for p in p3d_np.T])
            #print "kp_mask:,", kp_mask
            kp_filt = [kp for kp, m in zip(keypoints, kp_mask) if m]
            des_filt = descriptors[np.array(kp_mask)]
            p3d_np = p3d_np[:, kp_mask]
            p2d_np = p2d_np[:, kp_mask]
        else:
            kp_filt = []
            des_filt = descriptors
            p3d_np = np.array([])
            p2d_np = np.array([])
        return kp_filt, des_filt, p3d_np, p2d_np

    def match(self,des,oi):
        if self.settings['features_type'] == 'orb':
            matches = self.bf.match(des, oi.appearance.descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            matches_good = matches
        elif self.settings["features_type"] == 'sift':
            matches = self.bf.knnMatch(des, oi.appearance.descriptors, k=2)
            matches_good = []
            for m, n in matches:
                if m.distance < 0.85 * n.distance:
                    matches_good.append(m)

        print "Matched features:", len(matches_good)
        return matches_good

    def detect(self,imgs,clbs):
        depth = imgs[0]
        img = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
        if len(imgs)==3: mask = imgs[2]
        else: mask = np.ones_like(img)
        camera = clbs[0]

        kp, des = self.detector.detectAndCompute(img, mask)
        if self.settings['method'] == '3d3d':
            kp, des, p3d, p2d = self.filter_depth(kp, des, clbs[0], depth)
            #p3d, p2d = f2d.GetPointsFromKeypoints(kp, clbs[0], depth)
            #print "p3d:\n", p3d
        elif self.settings['method'] == '2d3d':
            p2d = f2d.ConvertKeypointsArray(kp)
        #print des.shape
        #p3d_def_vec = mbv.Core.Vector3fStorage(self.appearance.p3d_def.T)

        # BF Matching
        self.states = []
        #self.transformations = []
        self.inliers = []
        self.outliers = []
        for oi in self.objects_data:
            matches_good = self.match(des,oi)

            src_match_indices = np.array([m.trainIdx for m in matches_good])
            dst_match_indices = np.array([m.queryIdx for m in matches_good])

            if len(matches_good) > self.settings['min_matches']:
                src_pts_vec = mbv.Core.Vector3fStorage(oi.appearance.p3d_defpose[:,src_match_indices].T)
                dst_pts_vec = mbv.Core.Vector2fStorage(p2d[:,dst_match_indices].T)


                if self.settings['method'] == '3d3d':
                    src_pts_np = src_pts_vec.data.T
                    dst_pts_np = p3d[:, dst_match_indices]
                    Rt, outliers_idx = RigidObjectPosest3D.CalcRigidTransformRansac(src_pts_np, dst_pts_np, self.settings['ransac'])
                    #print 'Rt', Rt, outliers_idx
                    if np.isnan(Rt).any(): state_cur = None
                    else: state_cur = StateVectorTools.SetPositionRotationRt(oi.default_state, Rt)
                elif self.settings['method'] == '2d3d':
                    rtvec = mbv.Core.DoubleVector()
                    outliers_idx = mbv.Core.IntVector()
                    pm3d.posest(rtvec, dst_pts_vec, src_pts_vec, self.settings['inliers_ratio'], camera, outliers_idx)
                    outliers_idx = [i for i in outliers_idx]
                    state_cur = StateVectorTools.SetPositionRotationRvecT(oi.default_state, rtvec)

                self.states.append(state_cur)
                #self.transformations.append(rtvec)
                cur_inliers = [p for i,p in enumerate(dst_pts_vec) if i not in outliers_idx ]
                self.inliers.append(cur_inliers)
                cur_outliers = [p for i,p in enumerate(dst_pts_vec) if i in outliers_idx ]
                self.outliers.append(cur_outliers)

            else:
                self.states.append(None)
                #self.transformations.append(None)
                self.inliers.append([])
                self.outliers.append([])
        return self.states









