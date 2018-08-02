import BlenderMBV.BlenderMBVLib.AngleTransformations as at
import PythonModel3dTracker.Paths as Paths
from PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools import Visualizer
import PythonModel3dTracker.PyMBVAll as mbv
import cv2


def print_lanmark_info(l):
    print 'Landmark Name:',l.name, ', linked geom:', l.linked_geometry,
    print 'init pos:', l.pos,
           #', issue inst id:', l.primitiveid_pair.iss, ' / ',l.primitiveid_pair.ins



model_xml = Paths.model3d_dict['mh_bream_glbscl']['path']
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
landmark_names = mbv.Core.StringVector([b.key() for b in model3d.parts.bones_map])
#landmark_names, landmarks = M3DU.GetDefaultModelLandmarks(model3d,names_l)
landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names,
                                                         landmark_names,
                                                         mbv.PF.ReferenceFrame.RFGeomLocal,
                                                         mbv.Core.Vector3fStorage([mbv.Core.Vector3(0, 0, 0)]),
                                                         model3d.parts.bones_map)

transform_node = mbv.Dec.TransformNode()
mbv.PF.LoadTransformNode(model3d.transform_node_filename, transform_node)
landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks)


init_pos = model3d.default_state
value_range = model3d.high_bounds.data - model3d.low_bounds.data
visualizer = Visualizer(model3d, mmanager, decoder, renderer)
dims = []
steps = 12
for i in range(steps):
    # Setting param Vector
    init_pos[2] = 600#10 * f loat(i)
    rot_q = at.quaternion_from_euler(1.5,0+0.1*i,0)
    init_pos[3] = rot_q[1]
    init_pos[4] = rot_q[2]
    init_pos[5] = rot_q[3]
    init_pos[6] = rot_q[0]
    init_pos[7] = 0.036

    for d in dims:
         init_pos[d] = model3d.low_bounds[d] + i * value_range[d][0] / float(steps)

    print('state:', init_pos)

    #calculating new landmark_positions.
    landmark_positions = landmarks_decoder.decode(init_pos, landmarks)

    #Printing landmark info/new positions
    for l,p in zip(landmarks, landmark_positions):
        print_lanmark_info(l)
        print 'cur pos:', p.x,p.y,p.z


    #Rendering model
    viz = visualizer.visualize(init_pos,cam_meta,[landmark_positions])


    #Visualizing
    cv2.imshow("vizn",viz)
    key = chr(cv2.waitKey(0) & 255)
    if key == 'q': break



