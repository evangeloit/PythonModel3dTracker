import numpy as np
import PyOpenPose as OP
import os
import PyMBVCore as core

import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG
import PyCeresIK as IK
import copy
import cv2



class OpenPoseGrabber():
    landmark_names = {
         "COCO": [
             "Nose", "Neck",
             "RShoulder", "RElbow", "RWrist",
             "LShoulder", "LElbow", "LWrist",
             "RHip", "RKnee", "RAnkle",
             "LHip", "LKnee", "LAnkle",
             "REye", "LEye", "REar", "LEar"
         ]
    #     "COCO":["Nose","neck",
    #             "R.UArm","R.LArm","R.Wrist",
    #             "L.UArm","L.LArm","L.Wrist",
    #             "R.ULeg","R.LLeg","R.Foot",
    #             "L.ULeg","L.LLeg","L.Foot",
    #             "R.eye","L.eye","R.ear","L.ear"]
    }

    depth_diffs = {"COCO": [11,90,80,40,20,80,40,20,100,70,44,100,70,44,25,25,15,15]}
    def __init__(self,net_size=(320,240),res_size=(640,480),model_op = 'COCO',
                 model_op_path=None):
        assert model_op_path is not None
        self.model_op = model_op
        self.op = OP.OpenPose(net_size, (240, 240), res_size, model_op, model_op_path, 0, True)
        self.keypoints = None
        self.keypoints2d = None
        self.source = 'COCO'

    def seek(self,f): pass



    def acquire(self,images,calibs):
        depth = images[0]
        rgb = images[1]
        clb = calibs[0]
        self.op.detectPose(rgb)
        #hm = self.op.getHeatmaps()
        #parts = hm[:18]
        #background = hm[18]
        #cv2.imshow("background", background)
        persons = self.op.getKeypoints(self.op.KeypointType.POSE)[0]
        if persons is None:
            point_names = []
            self.keypoints = []
        else:
            # for i,p in enumerate(persons):
            #     for n, kp in zip(OpenPoseGrabber.landmark_names[self.model_op], p):
            #         print 'person', i, n, kp

            self.keypoints = []
            self.keypoints2d = []
            for p in persons:
                cur_keypoints, cur_keypoints2d = OpenPoseGrabber.PrepareRGBDKeypoints(p, depth, clb)
                self.keypoints.append(cur_keypoints)
                self.keypoints2d.append(cur_keypoints2d)
            #keypoints = OpenPoseGrabber.filterKeypointsDepth(keypoints, self.keypoints, 100)
            #print 'd_prev:', self.kp_depths
            #print 'd_new:', kp_depths
            #self.kp_depths = kp_depths

            point_names = OpenPoseGrabber.landmark_names[self.model_op]

        return point_names, self.keypoints, self.keypoints2d, clb, self.source


    @staticmethod
    def ConvertIK(keypoints_vec,clb):
        obsVecList = []
        if keypoints_vec is not None:
            for kv in keypoints_vec:
                keypoints = LG.LandmarksGrabber.pvec2np(kv).T.astype(np.float32)
                Kp = IK.Observations(IK.ObservationType.DEPTH, clb, keypoints)
                obsVec = IK.ObservationsVector([Kp])
                obsVecList.append(obsVec)
        return obsVecList

    @staticmethod
    def ConvertIK2D(keypoints_vec,clb):
        obsVecList = []
        if keypoints_vec is not None:
            for kv in keypoints_vec:
                keypoints = np.array(kv.__pythonize__()).astype(np.float32)
                Kp = IK.Observations(IK.ObservationType.COLOR, clb, keypoints)
                obsVec = IK.ObservationsVector([Kp])
                obsVecList.append(obsVec)
        return obsVecList


    @staticmethod
    def FilterKeypointsDepth(kp,kp_prev,t):
        if kp_prev is None: return kp
        counter = 0
        for p,pp in zip(kp,kp_prev):
            if abs(p.z - pp.z) > t:
                # print 'filtering ', p, 'prev:', pp
                p.z = pp.z
                counter += 1
            #else: dn = p.z
        # print 'filterKeypointsDepth filtered {0} keypoints'.format(counter)
        #print kp
        return kp

    @staticmethod
    def FilterKeypointsRandom(keypoints3d, keypoints2d, ratios=[0.1, 0.2]):
        #print ratios
        keypoints_out = core.Vector3fStorage(keypoints3d)
        n = len(keypoints3d)
        ratio2d = min(ratios[0],ratios[1])
        ratio3d = max(ratios[0], ratios[1])
        xclude_indices_3d = np.unique(np.random.choice(n, int(ratio3d*n), replace=True))
        if xclude_indices_3d.size == 0:
            xclude_indices_2d = xclude_indices_3d
        else:
            xclude_indices_2d = np.unique(np.random.choice(xclude_indices_3d, int(ratio2d*n), replace=True))
        for xind in xclude_indices_3d:
            keypoints_out[xind].x = keypoints2d[xind].x
            keypoints_out[xind].y = keypoints2d[xind].y
            keypoints_out[xind].z = 0
        for xind in xclude_indices_2d:
            keypoints_out[xind].x = 0
            keypoints_out[xind].y = 0
            keypoints_out[xind].z = 0
        # print 'Keypoints_out:\n', keypoints_out
        return keypoints_out

    @staticmethod
    def PrepareRGBDKeypoints(points2d_, depth, clb, w=4):
        n = points2d_.shape[0]
        points2d_ = points2d_[:,:2]


        points2d = core.Vector2fStorage()
        kp_depths = core.SingleVector()
        for i,(x, y) in enumerate(points2d_):
            p = core.Vector2(np.float(x),np.float(y))
            d = OpenPoseGrabber.getMedianDepth((x,y), depth, w)
            if d > 0: d += OpenPoseGrabber.depth_diffs["COCO"][i]
            points2d.append(p)
            kp_depths.append(d)
        points3d = clb.unproject(points2d, kp_depths)
        # Keeping 2d coordinates for points with zero depth.
        for p2d,p3d in zip(points2d,points3d):
            if p3d.z == 0:
                p3d.x = p2d.x
                p3d.y = p2d.y
        return points3d, points2d

    @staticmethod
    def getMedianDepth(p,depthmap,w=4):
        x = p[0]
        y = p[1]
        height = depthmap.shape[0]
        width = depthmap.shape[1]
        if (x > 0) and (y > 0) and (x < width) and (y < height):
            d = np.median(depthmap[max(y-w, 0):min(y+w, height-1),
                                   max(x-w, 0):min(x+w, width-1)])
            #d = np.nan_to_num(d)
        else:
            d = 0
        return d





