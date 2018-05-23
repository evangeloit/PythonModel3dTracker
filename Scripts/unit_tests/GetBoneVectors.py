import numpy as np
import copy
import BlenderMBV.BlenderMBVLib.AngleTransformations as at
import PythonModel3dTracker.Paths as Paths
from PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools import Visualizer
import PythonModel3dTracker.PyMBVAll as mbv
import cv2

import PythonModel3dTracker.PythonModelTracker.Model3dUtils as M3DU



model_xml = Paths.model3d_dict['mh_body_male_custom_vector']['path']
model3d = mbv.PF.Model3dMeta.create(str(model_xml))
model_parts = model3d.parts

print('Loaded model from <', model_xml, '>', ', bones:', model3d.n_bones, ', dims:', model3d.n_dims)

mmanager = mbv.Core.MeshManager()
decoder = model3d.createDecoder()
model3d.setupMeshManager(mmanager)
decoder.loadMeshTickets(mmanager)

n_bones = model3d.n_bones
model_parts.genBonesMap()
print('Parts Map:', model_parts.parts_map)
print('Bones Map:', model_parts.bones_map)


# Setting camera/renderer
cam_meta = mbv.Lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
renderer = mbv.Ren.RendererOGLCudaExposed.get()


#Creating Landmark3dInfo structs.
state = model3d.default_state
default_decoding = decoder.quickDecode(state)
landmarks_decoder = mbv.PF.LandmarksDecoder()
landmarks_decoder.decoder = decoder
bone_names = mbv.Core.StringVector([b.key() for b in model3d.parts.bones_map])
landmark_names_0 = mbv.Core.StringVector([b.key()+'_0' for b in model3d.parts.bones_map])
landmark_names_1 = mbv.Core.StringVector([b.key()+'_1' for b in model3d.parts.bones_map])
#landmark_names, landmarks = M3DU.GetDefaultModelLandmarks(model3d,names_l)
landmarks_0 = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names_0,
                                                         bone_names,
                                                         mbv.PF.ReferenceFrame.RFGeomLocal,
                                                         mbv.Core.Vector3fStorage([mbv.Core.Vector3(0, 0, 0)]),
                                                         model3d.parts.bones_map)
landmarks_1 = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names_1,
                                                         bone_names,
                                                         mbv.PF.ReferenceFrame.RFGeomLocal,
                                                         mbv.Core.Vector3fStorage([mbv.Core.Vector3(0, 1, 0)]),
                                                         model3d.parts.bones_map)


transform_node = mbv.Dec.TransformNode()
mbv.PF.LoadTransformNode(model3d.transform_node_filename, transform_node)
landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks_0)
landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks_1)


init_pos = model3d.default_state
value_range = model3d.high_bounds.data - model3d.low_bounds.data
visualizer = Visualizer(model3d, mmanager, decoder, renderer)
dims = [9,10]
steps = 4
for i in range(steps):
    # Setting param Vector
    init_pos[2] = 2700#10 * f loat(i)
    rot_q = at.quaternion_from_euler(1.5,0+0.1*i,0)
    init_pos[3] = rot_q[1]
    init_pos[4] = rot_q[2]
    init_pos[5] = rot_q[3]
    init_pos[6] = rot_q[0]


    for d in dims:
         init_pos[d] = model3d.low_bounds[d] + i * value_range[d][0] / float(steps)

    print 'state:', init_pos

    #calculating new landmark_positions.
    landmark_positions_0 = landmarks_decoder.decode(init_pos, landmarks_0)
    landmark_positions_1 = landmarks_decoder.decode(init_pos, landmarks_1)

    for b,l0,l1 in zip(bone_names, landmark_positions_0, landmark_positions_1):
        bone_vec = l1 - l0
        print b, bone_vec
        if b == 'R.UArm': vec_uarm = copy.deepcopy(bone_vec)
        if b == 'R.LArm': vec_larm = copy.deepcopy(bone_vec)
        if b == 'root': vec_root = copy.deepcopy(bone_vec)

    print 'angle ruarm, root', vec_uarm, vec_root, mbv.Core.glm.angle(vec_uarm, vec_root)
    print 'angle rlarm, ruarm', vec_uarm, vec_larm, mbv.Core.glm.angle(vec_uarm, vec_larm)
    print 'angle rlarm, root', vec_larm, vec_root, mbv.Core.glm.angle(vec_root, vec_larm)

    #Rendering model
    viz = visualizer.visualize(init_pos,cam_meta,[landmark_positions_0, landmark_positions_1])


    #Visualizing
    cv2.imshow("viz",viz)
    key = chr(cv2.waitKey(0) & 255)
    if key == 'q': break



