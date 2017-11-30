import PyMBVCore as core
import PyMBVAcquisition as acq
import PyMBVParticleFilter as mpf
import PyMBVDecoding as dec
import PyModel3dTracker as mt
import cv2
import csv
import numpy as np
import os.path

primitives_dict = {
    "human_ext":
    {"head":"head_sphere_0","neck":"neck_sphere_0","bodyCenter":"body_sphere_1",
     "hip":"body_sphere_2","leftShoulder":"arm_left_sphere_0","rightShoulder":"arm_right_sphere_0",
     "leftElbow":"arm_left_sphere_1","rightElbow":"arm_right_sphere_1","leftWrist":"arm_left_sphere_2",
     "rightWrist":"arm_right_sphere_2","leftLegRoot":"leg_left_sphere_0",
     "rightLegRoot":"leg_right_sphere_0","leftKnee":"leg_left_sphere_1","rightKnee":"leg_right_sphere_1",
     "leftAnkle":"leg_left_sphere_2","rightAnkle":"leg_right_sphere_2"},
    "mh_body_male":
    {"head":"head","neck":"neck","bodyCenter":"spine-1",
     "hip":"hips","leftShoulder":"deltoid.L","rightShoulder":"deltoid.R",
     "leftElbow":"forearm.L","rightElbow":"forearm.R","leftWrist":"hand.L",
     "rightWrist":"hand.R","leftLegRoot":"thigh.L",
     "rightLegRoot":"thigh.R","leftKnee":"shin.L","rightKnee":"shin.R",
     "leftAnkle":"foot.L","rightAnkle":"foot.R"},
    "hand_skinned_rds":
    {'f_pinky.03.R':'f_pinky.03.R',
     'f_middle.03.R':'f_middle.03.R',
     'f_ring.03.R':'f_ring.03.R',
     'thumb.03.R':'thumb.03.R',
     'f_index.03.R':'f_index.03.R'},
    "mh_body_male_custom":
    {'L.LLeg': 'L.LLeg', 'L.ULeg': 'L.ULeg', 'R.LLeg': 'R.LLeg', 'R.Foot': 'R.Foot', 'R.LArm': 'R.LArm',
     'R.eye': 'R.eye', 'L.shoulder': 'L.shoulder', 'R.torso': 'R.torso', 'L.LArm': 'L.LArm',
     'R.shoulder': 'R.shoulder', 'L.Wrist': 'L.Wrist', 'R.ULeg': 'R.ULeg', 'R.ear': 'R.ear', 'L.ear': 'L.ear',
     'L.eye': 'L.eye', 'Nose': 'Nose', 'L.UArm': 'L.UArm', 'neck': 'neck', 'neck.001': 'neck.001', 'root': 'root',
     'R.Wrist': 'R.Wrist', 'L.torso': 'L.torso', 'R.UArm': 'R.UArm', 'L.Foot': 'L.Foot'}
    #{'L.LLeg': 15, 'L.ULeg': 14, 'R.LLeg': 11, 'R.Foot': 12, 'R.LArm': 3, 'R.eye': 21, 'L.shoulder': 5, 'R.torso': 9,
    # 'L.LArm': 7, 'R.shoulder': 1, 'L.Wrist': 8, 'R.ULeg': 10, 'R.ear': 23, 'L.ear': 22, 'L.eye': 20, 'Nose': 19,
    # 'L.UArm': 6, 'neck': 17, 'neck.001': 18, 'root': 0, 'R.Wrist': 4, 'L.torso': 13, 'R.UArm': 2, 'L.Foot': 16}
}
primitives_dict["human_ext_collisions"] = primitives_dict["human_ext"]
primitives_dict["mh_body_male_meta"] = primitives_dict["mh_body_male"]
primitives_dict["mh_body_male_meta_grpscl"] = primitives_dict["mh_body_male"]
primitives_dict["mh_body_male_custom_meta"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_0850"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_0900"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_0950"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_1050"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_1100"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_1150"] = primitives_dict["mh_body_male_custom"]
primitives_dict["mh_body_male_custom_meta_glbscl"] = primitives_dict["mh_body_male_custom"]



            

class LandmarksGrabber:
    """
    Attributes:
        file_format
        f_count
        landmarks_filename
        clb_filename
        clb        
    """
    supported_formats = ['damien','bvh','roditak']
    
    def __init__(self, file_format, landmarks_filename, clb_filename, model_name):
        assert file_format in LandmarksGrabber.supported_formats
        self.file_format = file_format
        self.landmarks_filename = landmarks_filename

        self.clb = LandmarksGrabber.loadCalibTxt(clb_filename)
        self.t, extr_rot = self.clb.camera.OpenCV_getExtrinsics()
        self.R, _ = cv2.Rodrigues(extr_rot)
        self.model_name = model_name
        self.f_count = 0
        self.bvh_point_names = None
        self.bvh_points = None
        self.fps = 30.
        self.points_vec = core.Vector3fStorage()
        self.point_names = []
        if self.file_format == 'bvh':
            assert os.path.isfile(landmarks_filename)
            self.bvh_point_names, self.bvh_points, self.fps = mt.LoadBvh(str(landmarks_filename), 'x y z')

    @staticmethod
    def getPrimitiveNamesfromLandmarkNames(ldm_names, model_name):
        primitives = core.StringVector()
        for p in ldm_names:
            primitives.append(primitives_dict[model_name][p])
        return primitives

    def seek(self,f):
        self.f_count = f

    def acquire(self,images=None,calibs=None):
        if self.file_format == 'damien':
            ldm_filename = self.landmarks_filename % self.f_count
            if os.path.isfile(ldm_filename):
                self.point_names, points = LandmarksGrabber.loadDamienLandmarks(ldm_filename)
                points = points.transpose()
            else: points = LandmarksGrabber.pvec2np(self.points_vec)
        elif self.file_format == 'roditak':
            ldm_filename = self.landmarks_filename % self.f_count
            #print(ldm_filename)
            assert os.path.isfile(ldm_filename)

            self.point_names, points = LandmarksGrabber.loadRoditakLandmarks(ldm_filename)
            points = points.transpose()
        elif self.file_format == 'bvh':
            bvh_frame = int(round(self.f_count*self.fps/30.))
            self.point_names = self.bvh_point_names[bvh_frame]
            points = LandmarksGrabber.pvec2np(self.bvh_points[bvh_frame])

        points = np.dot(self.R,points) + self.t

        self.points_vec = LandmarksGrabber.np2pvec(points)

        self.f_count += 1
        return self.point_names, self.points_vec, self.clb

    @staticmethod
    def np2pvec(points_np):
        points_mbv = core.Vector3fStorage()
        points_np = points_np.transpose()
        for i, p in enumerate(points_np):
            point_mbv = core.Vector3()
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
            # print core.Vector3(p[0],p[1],p[2])
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

        clb = core.CameraMeta()
        cam = core.CameraFrustum()
        clb.width = imwidth
        clb.height = imheight
        C = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], imwidth, imheight)
        cam.setIntrinsics(C[0], C[1], C[2], C[3], C[4], C[5], 400, 10000)
        cam.OpenCV_setExtrinsics(tr, rot)
        clb.camera = cam
        return clb



def GetDefaultModelLandmarks(model3d, landmark_names=None):
    # pf.Landmark3dInfoVec()
    if landmark_names is None:
        landmark_names = model3d.parts.parts_map['all']
    if model3d.model_type == mpf.Model3dType.Primitives:
        landmarks = mpf.Landmark3dInfoPrimitives.create_multiple(landmark_names,
                                                                landmark_names,
                                                                mpf.ReferenceFrame.RFGeomLocal,
                                                                core.Vector3fStorage([core.Vector3(0, 0, 0)]),
                                                                model3d.parts.primitives_map)
    else:
        landmarks = mpf.Landmark3dInfoSkinned.create_multiple(landmark_names,
                                                              landmark_names,
                                                             mpf.ReferenceFrame.RFGeomLocal,
                                                             core.Vector3fStorage([core.Vector3(0, 0, 0)]),
                                                             model3d.parts.bones_map)
        #print(landmark_names)
        transform_node = dec.TransformNode()
        mpf.LoadTransformNode(model3d.transform_node_filename, transform_node)
        landmarks_decoder = mpf.LandmarksDecoder()
        landmarks_decoder.convertReferenceFrame(mpf.ReferenceFrame.RFModel, transform_node, landmarks)

    return landmarks



    