import PythonModel3dTracker.PyMBVAll as mbv

import PyModel3dTracker as mt
import cv2
import csv
import numpy as np
import os.path
from PythonModel3dTracker.PythonModelTracker.LandmarksCorrespondences import primitives_dict
from PythonModel3dTracker.PythonModelTracker.LandmarksCorrespondences import model_landmark_positions


class LandmarksGrabber:
    """
    Attributes:
        source
        f_count
        landmarks_filename
        clb_filename
        clb        
    """
    supported_formats = ['damien','bvh','roditak','csv']
    
    def __init__(self, source, landmarks_filename, clb_filename, model_name = None):
        assert source in LandmarksGrabber.supported_formats
        self.source = source
        self.landmarks_filename = landmarks_filename

        self.clb = LandmarksGrabber.loadCalibTxt(clb_filename)
        self.t, extr_rot = self.clb.camera.OpenCV_getExtrinsics()
        self.R, _ = cv2.Rodrigues(extr_rot)
        self.model_name = model_name
        self.f_count = 0
        self.preloaded_point_names = None
        self.preloaded_points = None
        self.fps = 30.
        self.points_vec = mbv.Core.Vector3fStorage()
        self.point_names = []
        self.filter_landmarks = False
        if self.source == 'bvh':
            assert os.path.isfile(landmarks_filename)
            self.preloaded_point_names, self.preloaded_points, self.fps = mt.LoadBvh(str(landmarks_filename), 'x y z')
        if self.source == 'csv':
            assert os.path.isfile(landmarks_filename)
            self.preloaded_point_names, self.preloaded_points = LandmarksGrabber.loadCsvLandmarks(str(landmarks_filename))

    @staticmethod
    def GetFilteredLandmarks(model_name, ldm_obs_source, ldm_obs_names, ldm_obs):
        prim_dict = primitives_dict[(str(ldm_obs_source), str(model_name))]
        lnames_cor = [l for l in prim_dict]

        ldm_obs_names = [l for l in ldm_obs_names]
        idx_obs = [ldm_obs_names.index(g) for g in lnames_cor]

        names_obs_cor = [ldm_obs_names[l] for l in idx_obs]
        ldm_obs_cor = mbv.Core.Vector3fStorage([ldm_obs[l] for l in idx_obs])
        return names_obs_cor, ldm_obs_cor

    @staticmethod
    def getPrimitiveNamesfromLandmarkNames(ldm_names,landmark_source,model_name):
        primitives = mbv.Core.StringVector()
        for p in ldm_names:
            if p in primitives_dict[(str(landmark_source), str(model_name))]:
                primitives.append(primitives_dict[(str(landmark_source), str(model_name))][p])
            else:
                primitives.append("None")
        return primitives

    def seek(self,f):
        self.f_count = f

    def acquire(self,images=None,calibs=None):
        if self.source == 'damien':
            ldm_filename = self.landmarks_filename % self.f_count
            if os.path.isfile(ldm_filename):
                self.point_names, points = LandmarksGrabber.loadDamienLandmarks(ldm_filename)
                points = points.transpose()
            else: points = LandmarksGrabber.pvec2np(self.points_vec)
        elif self.source == 'roditak':
            ldm_filename = self.landmarks_filename % self.f_count
            #print(ldm_filename)
            assert os.path.isfile(ldm_filename)

            self.point_names, points = LandmarksGrabber.loadRoditakLandmarks(ldm_filename)
            points = points.transpose()
        elif self.source == 'bvh':
            bvh_frame = int(round(self.f_count*self.fps/30.))
            self.point_names = self.preloaded_point_names[bvh_frame]
            points = LandmarksGrabber.pvec2np(self.preloaded_points[bvh_frame])
        elif self.source == 'csv':
            self.point_names = self.preloaded_point_names
            points = self.preloaded_points[self.f_count]

        points = np.dot(self.R,points) + self.t

        self.points_vec = LandmarksGrabber.np2pvec(points)


        self.f_count += 1

        if self.filter_landmarks:
            self.point_names, self.points_vec = \
                LandmarksGrabber.GetFilteredLandmarks(self.model_name, self.source, self.point_names, self.points_vec)


        self.points2d_vec = calibs[0].project(mbv.Core.Vector3fStorage(self.points_vec))

        return self.point_names, [self.points_vec], [ self.points2d_vec ], self.clb, self.source


    @staticmethod
    def np2pvec(points_np):
        points_mbv = mbv.Core.Vector3fStorage()
        points_np = points_np.transpose()
        for i, p in enumerate(points_np):
            point_mbv = mbv.Core.Vector3()
            point_mbv.x = np.float(p[0])
            point_mbv.y = np.float(p[1])
            point_mbv.z = np.float(p[2])
            points_mbv.append(point_mbv)
        return points_mbv

    @staticmethod
    def pvec2np(points_mbv):
        points_np = np.zeros((3, len(points_mbv)))
        for p, point in enumerate(points_mbv):
            points_np[0, p] = point.x
            points_np[1, p] = point.y
            points_np[2, p] = point.z
        return points_np


    @staticmethod
    def apply_transform(points_mbv, R,t):
        points = np.zeros((3, len(points_mbv)))
        for p, point in enumerate(points_mbv):
            points[0, p] = point.x
            points[1, p] = point.y
            points[2, p] = point.z
        points = np.dot(R, points) + t
        points = points.transpose()
        for i, p in enumerate(points):
            # print mbv.Core.Vector3(p[0],p[1],p[2])
            points_mbv[i].x = p[0]
            points_mbv[i].y = p[1]
            points_mbv[i].z = p[2]
        return points_mbv

    @staticmethod
    def loadDamienLandmarks(ldm_filename):
        with open(ldm_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            points = np.empty((0, 3), float)
            point_names = []
            for i, row in enumerate(reader):
                if i == 0:
                    n_points = int(row[0])
                if ((i >= 1) and (i <= n_points)):
                    if (int(row[4]) == 1):
                        point_names.append(row[0])
                        points = np.append(points, np.array([[float(row[1]), float(row[2]), float(row[3])]]), axis=0)
        return point_names, points


    @staticmethod
    def loadRoditakLandmarks(ldm_filename):
        with open(ldm_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            points = np.empty((0, 3), float)
            point_names = ['f_pinky.03.R','f_middle.03.R','f_ring.03.R','thumb.03.R','f_index.03.R']
            for i, row in enumerate(reader):
                points = np.append(points, np.array([[float(row[0]), float(row[1]), float(row[2])]]), axis=0)
        return point_names, points



    @staticmethod
    def loadCsvLandmarks(ldm_filename):
        with open(ldm_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')

            points = []
            for i, row in enumerate(reader):
                N = len(row)
                if i == 2: point_names = [row[j][:-2] for j in range(N) if (((j - 2) % 4) == 0)]
                if i>2:
                    points_cur = np.empty((0, 3), float)
                    for j in range(N):
                        if ((j - 2) % 4) == 0:
                            points_cur = np.append(points_cur, np.array([[float(row[j]), float(row[j+1]),
                                                                          float(row[j+2])]]), axis=0)
                    points_cur = points_cur.transpose()
                    points.append(points_cur)
        return point_names, points

    @staticmethod
    def loadCalibTxt(clb_filename):
        with open(clb_filename) as txtfile:
            x = txtfile.read().splitlines()
            # print x
            for i, item in enumerate(x):
                tokens = item.split('=')
                if (tokens[0] == '%ImageWidth'):
                    imwidth = int(tokens[1])
                if (tokens[0] == '%ImageHeight'):
                    imheight = int(tokens[1])
                if (item == '%I'):
                    K = np.array([[float(x[i + 1]), float(x[i + 2]), float(x[i + 3])],
                                  [float(x[i + 4]), float(x[i + 5]), float(x[i + 6])],
                                  [0., 0., 1.]])
                if (item == '%T'):
                    tr = np.array([[float(x[i + 1])], [float(x[i + 2])], [float(x[i + 3])]])
                    # print tr
                if (item == '%R'):
                    rot = np.array([[float(x[i + 1])], [float(x[i + 2])], [float(x[i + 3])]])
                    # print rot

        clb = mbv.Core.CameraMeta()
        cam = mbv.Core.CameraFrustum()
        clb.width = imwidth
        clb.height = imheight
        C = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], imwidth, imheight)
        cam.setIntrinsics(C[0], C[1], C[2], C[3], C[4], C[5], 400, 10000)
        cam.OpenCV_setExtrinsics(tr, rot)
        clb.camera = cam
        return clb










    