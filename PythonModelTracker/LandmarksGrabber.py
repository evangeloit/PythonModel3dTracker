import PyMBVCore as core
import PyMBVAcquisition as acq
import PyMBVParticleFilter as mpf
import PyMBVDecoding as dec
import PyModel3dTracker as mt
import cv2
import csv
import numpy as np
import os.path

# Dictionary of landmark correspondences between: (landmard_detection_source, skinned_model).
primitives_dict = {
    ("damien", "human_ext"):
    {"head":"head_sphere_0","neck":"neck_sphere_0","bodyCenter":"body_sphere_1",
     "hip":"body_sphere_2","leftShoulder":"arm_left_sphere_0","rightShoulder":"arm_right_sphere_0",
     "leftElbow":"arm_left_sphere_1","rightElbow":"arm_right_sphere_1","leftWrist":"arm_left_sphere_2",
     "rightWrist":"arm_right_sphere_2","leftLegRoot":"leg_left_sphere_0",
     "rightLegRoot":"leg_right_sphere_0","leftKnee":"leg_left_sphere_1","rightKnee":"leg_right_sphere_1",
     "leftAnkle":"leg_left_sphere_2","rightAnkle":"leg_right_sphere_2"},
    ("damien", "mh_body_male"):
    {"head":"head","neck":"neck","bodyCenter":"spine-1",
     "hip":"hips","leftShoulder":"deltoid.L","rightShoulder":"deltoid.R",
     "leftElbow":"forearm.L","rightElbow":"forearm.R","leftWrist":"hand.L",
     "rightWrist":"hand.R","leftLegRoot":"thigh.L",
     "rightLegRoot":"thigh.R","leftKnee":"shin.L","rightKnee":"shin.R",
     "leftAnkle":"foot.L","rightAnkle":"foot.R"},
    ("roditak", "hand_skinned_rds"):
    {'f_pinky.03.R':'f_pinky.03.R',
     'f_middle.03.R':'f_middle.03.R',
     'f_ring.03.R':'f_ring.03.R',
     'thumb.03.R':'thumb.03.R',
     'f_index.03.R':'f_index.03.R'},
    ("coco", "mh_body_male_custom"):
    {'L.LLeg': 'L.LLeg', 'L.ULeg': 'L.ULeg', 'R.LLeg': 'R.LLeg', 'R.Foot': 'R.Foot', 'R.LArm': 'R.LArm',
     'R.eye': 'R.eye', 'L.shoulder': 'L.shoulder', 'R.torso': 'R.torso', 'L.LArm': 'L.LArm',
     'R.shoulder': 'R.shoulder', 'L.Wrist': 'L.Wrist', 'R.ULeg': 'R.ULeg', 'R.ear': 'R.ear', 'L.ear': 'L.ear',
     'L.eye': 'L.eye', 'Nose': 'Nose', 'L.UArm': 'L.UArm', 'neck': 'neck', 'neck.001': 'neck.001', 'root': 'root',
     'R.Wrist': 'R.Wrist', 'L.torso': 'L.torso', 'R.UArm': 'R.UArm', 'L.Foot': 'L.Foot'},
    ("bvh", "mh_body_male_custom"):
        {'LeftUpLeg': 'L.torso', 'LeftLeg': 'L.ULeg', 'LeftFoot': 'L.LLeg',
         'RightUpLeg': 'R.torso', 'RightLeg': 'R.ULeg', 'RightFoot': 'R.LLeg',
         'LeftArm': 'L.shoulder', 'LeftForeArm': 'L.UArm', 'LeftHand': 'L.Wrist',
         'RightArm': 'R.shoulder', 'RightForeArm': 'R.UArm', 'RightHand': 'R.Wrist',
         'Neck': 'neck.001', 'spine': 'root'
        },
    ("csv", "mh_body_male_custom"):
    {"head":"neck.001","neck":"neck","bodyCenter":"root",
     "leftShoulder":"L.UArm","rightShoulder":"R.UArm",
     "leftElbow":"L.LArm","rightElbow":"R.LArm","leftWrist":"L.Wrist",
     "rightWrist":"R.Wrist","leftLegRoot":"L.ULeg",
     "rightLegRoot":"R.ULeg","leftKnee":"L.LLeg","rightKnee":"R.LLeg",
     "leftAnkle":"L.Foot","rightAnkle":"R.Foot"},


    # {'LeftLeg': 'L.LLeg', 'LeftUpLeg': 'L.ULeg', 'RightLeg': 'R.LLeg', 'RightFoot': 'R.Foot', 'RightForeArm': 'R.LArm',
    #  'LeftShoulder': 'L.shoulder', 'LeftForeArm': 'L.LArm',
    #  'RightShoulder': 'R.shoulder', 'LeftHand': 'L.Wrist', 'RightUpLeg': 'R.ULeg',
    #   'Head': 'neck.001', 'LeftArm': 'L.UArm', 'Neck': 'neck.001', 'spine': 'root',
    #  'RightHand': 'R.Wrist', 'RightArm': 'R.UArm', 'LeftFoot': 'L.Foot'}

    # 'Hips', 'spine', 'spine1', 'spine2',
    # 'Neck', 'Head', 'Site', 'RightShoulder',
    # 'RightArm', 'RightArmRoll', 'RightForeArm',
    # 'RightForeArmRoll', 'RightHand', 'Site', 'LeftShoulder',
    # 'LeftArm', 'LeftArmRoll', 'LeftForeArm', 'LeftForeArmRoll',
    # 'LeftHand', 'Site', 'RightUpLeg', 'RightUpLegRoll', 'RightLeg',
    # 'RightLegRoll', 'RightFoot', 'RightToeBase', 'Site', 'LeftUpLeg',
    # 'LeftUpLegRoll', 'LeftLeg', 'LeftLegRoll', 'LeftFoot', 'LeftToeBase', 'Site'

# {'LeftLeg': 'L.LLeg', 'LeftUpLeg': 'L.ULeg', 'RightLeg': 'R.LLeg', 'RightFoot': 'R.Foot', 'RightForeArm': 'R.LArm',
#      'Head': 'R.eye', 'LeftShoulder': 'L.shoulder', 'spine2': 'R.torso', 'LeftForeArm': 'L.LArm',
#      'RightShoulder': 'R.shoulder', 'LeftHand': 'L.Wrist', 'RightUpLeg': 'R.ULeg', 'Head': 'R.ear', 'Head': 'L.ear',
#      'Head': 'L.eye', 'Head': 'Nose', 'LeftArm': 'L.UArm', 'Neck': 'neck', 'Neck': 'neck.001', 'spine': 'root',
#      'RightHand': 'R.Wrist', 'spine2': 'L.torso', 'RightArm': 'R.UArm', 'LeftFoot': 'L.Foot'}
}
primitives_dict[("damien", "human_ext_collisions")] = primitives_dict[("damien", "human_ext")]
primitives_dict[("damien", "mh_body_male_meta")] = primitives_dict[("damien", "mh_body_male")]
primitives_dict[("damien", "mh_body_male_meta_grpscl")] = primitives_dict[("damien", "mh_body_male")]
primitives_dict[("coco", "mh_body_male_custom_meta")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_0850")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_0900")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_0950")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_1050")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_1100")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_1150")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("coco", "mh_body_male_custom_meta_glbscl")] = primitives_dict[("coco", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_meta")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_0850")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_0900")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_0950")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_1050")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_1100")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_1150")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("csv", "mh_body_male_custom_0950")] = primitives_dict[("csv", "mh_body_male_custom")]


            

class LandmarksGrabber:
    """
    Attributes:
        file_format
        f_count
        landmarks_filename
        clb_filename
        clb        
    """
    supported_formats = ['damien','bvh','roditak','csv']
    
    def __init__(self, file_format, landmarks_filename, clb_filename, model_name = None):
        assert file_format in LandmarksGrabber.supported_formats
        self.file_format = file_format
        self.landmarks_filename = landmarks_filename

        self.clb = LandmarksGrabber.loadCalibTxt(clb_filename)
        self.t, extr_rot = self.clb.camera.OpenCV_getExtrinsics()
        self.R, _ = cv2.Rodrigues(extr_rot)
        self.model_name = model_name
        self.f_count = 0
        self.preloaded_point_names = None
        self.preloaded_points = None
        self.fps = 30.
        self.points_vec = core.Vector3fStorage()
        self.point_names = []
        if self.file_format == 'bvh':
            assert os.path.isfile(landmarks_filename)
            self.preloaded_point_names, self.preloaded_points, self.fps = mt.LoadBvh(str(landmarks_filename), 'x y z')
        if self.file_format == 'csv':
            assert os.path.isfile(landmarks_filename)
            self.preloaded_point_names, self.preloaded_points = LandmarksGrabber.loadCsvLandmarks(str(landmarks_filename))

    @staticmethod
    def getPrimitiveNamesfromLandmarkNames(ldm_names,landmark_source,model_name):
        primitives = core.StringVector()
        for p in ldm_names:
            if p in primitives_dict[(str(landmark_source), str(model_name))]:
                primitives.append(primitives_dict[(str(landmark_source), str(model_name))][p])
            else:
                primitives.append("None")
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
            self.point_names = self.preloaded_point_names[bvh_frame]
            points = LandmarksGrabber.pvec2np(self.preloaded_points[bvh_frame])
        elif self.file_format == 'csv':
            self.point_names = self.preloaded_point_names
            points = self.preloaded_points[self.f_count]

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

    return landmark_names, landmarks


def GetCorrespondingLandmarks(model_name, ldm_model_names, ldm_model, ldm_obs_source, ldm_obs_names, ldm_obs):
    lnames_cor = LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(ldm_obs_names, ldm_obs_source, model_name)
    idx_obs = [i for i, g in enumerate(lnames_cor) if g != 'None']
    idx_model = [ldm_model_names.index(g) for g in lnames_cor if g != 'None']

    names_model_cor = [ldm_model_names[l] for l in idx_model]
    ldm_model_cor = [ldm_model[l] for l in idx_model]
    names_obs_cor = [ldm_obs_names[l] for l in idx_obs]
    ldm_obs_cor = [ [float(ldm_obs[l].data[0, 0]), float(ldm_obs[l].data[1, 0]),
                     float(ldm_obs[l].data[2, 0])] for l in idx_obs]
    return names_model_cor, ldm_model_cor, names_obs_cor, ldm_obs_cor



    