import itertools
import PythonModel3dTracker.PyMBVAll as mbv
from PythonModel3dTracker.PythonModelTracker.Landmarks.Model3dLandmarks import GetDefaultModelLandmarks, GetLandmarkPos

class BoneGeometry:
    def __init__(self, model3d, decoder):
        self.model3d = model3d
        self.decoder = decoder
        self.landmarks_decoder = mbv.PF.LandmarksDecoder()
        self.landmarks_decoder.decoder = decoder
        self.all_bone_names = mbv.Core.StringVector([b.key() for b in self.model3d.parts.bones_map])
        self.all_landmarks_0, self.all_landmarks_1 = self.getBoneLandmarks(self.all_bone_names)



    def getBoneLandmarks(self, bone_names = None):
        if bone_names is None: bone_names = self.all_bone_names
        else: bone_names = mbv.Core.StringVector(bone_names)
        landmark_names_0 = mbv.Core.StringVector([b + '_0' for b in bone_names])
        landmark_names_1 = mbv.Core.StringVector([b + '_1' for b in bone_names])
        landmarks_0 = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names_0,
                                                                            bone_names,
                                                                            mbv.PF.ReferenceFrame.RFGeomLocal,
                                                                            mbv.Core.Vector3fStorage(
                                                                                [mbv.Core.Vector3(0, 0, 0)]),
                                                                            self.model3d.parts.bones_map)
        landmarks_1 = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names_1,
                                                                            bone_names,
                                                                            mbv.PF.ReferenceFrame.RFGeomLocal,
                                                                            mbv.Core.Vector3fStorage(
                                                                                [mbv.Core.Vector3(0, 1, 0)]),
                                                                            self.model3d.parts.bones_map)
        transform_node = self.decoder.kinematics
        self.landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks_0)
        self.landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks_1)
        return landmarks_0, landmarks_1


    def calcVectors(self, state, bone_names = None):
        if bone_names is None:
            bone_names = self.all_bone_names
            landmarks_0 = self.all_landmarks_0
            landmarks_1 = self.all_landmarks_1
        else:
            landmarks_0, landmarks_1 = self.getBoneLandmarks(bone_names)

        landmark_positions_0 = self.landmarks_decoder.decode(state, landmarks_0)
        landmark_positions_1 = self.landmarks_decoder.decode(state, landmarks_1)
        bone_vectors = {}
        for b, l0, l1 in zip(bone_names, landmark_positions_0, landmark_positions_1):
            bone_vectors[b] = l1 - l0
        return bone_vectors

    def calcAngles(self, state, bone_vectors = None, bone_name_pairs = None):
        if bone_vectors is None: bone_vectors = self.calcVectors(state)
        if bone_name_pairs is None: bone_name_pairs = itertools.product(bone_vectors, bone_vectors)
        bone_angles = {}
        for b1, b2 in bone_name_pairs:
            if b1 in bone_vectors and b2 in bone_vectors:
                bone_angles[ (b1, b2) ] = mbv.Core.glm.angle(bone_vectors[b1], bone_vectors[b2])
        return bone_angles




