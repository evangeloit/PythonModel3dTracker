import PyMBVRendering as ren
import matplotlib.pyplot as plt
import numpy as np

import BlenderMBVLib.AngleTransformations as at
import BlenderMBVLib.RenderingUtils as ru
import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVLibraries as lib
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as pf
import cv2

import PythonModelTracker.LandmarksGrabber


def print_lanmark_info(l):
    print('Landmark Name:',l.name, ', linked geom:', l.linked_geometry)
    print('init pos:', l.pos.data[0],l.pos.data[1],l.pos.data[2])
           #', issue inst id:', l.primitiveid_pair.iss, ' / ',l.primitiveid_pair.ins



model_xml = Paths.model3d_dict['mh_body_male_meta'][1]
model3d = pf.Model3dMeta.create(model_xml)
model_parts = model3d.parts

print('Loaded model from <', model_xml, '>', ', bones:', model3d.n_bones, ', dims:', model3d.n_dims)

mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
decoder = model3d.createDecoder()
model3d.setupMeshManager(mmanager)
decoder.loadMeshTickets(mmanager)

if model3d.model_type == pf.Model3dType.Skinned:
    n_bones = model3d.n_bones
    model_parts.genBonesMap()
    print('Parts Map:', model_parts.parts_map)
    print('Bones Map:', model_parts.bones_map)
else:
    decoder = dec.GenericDecoder()
    decoder.loadFromFile(model3d.model_collada,False)
    model_parts.genPrimitivesMap(decoder)
    n_bones = 0



# Setting camera/renderer
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
renderer = ren.RendererOGLCudaExposed.get()


#Creating Landmark3dInfo structs.
state = model3d.default_state
default_decoding = decoder.quickDecode(state)
landmarks_decoder = pf.LandmarksDecoder()
names_l = core.StringVector([b.key() for b in model3d.parts.bones_map])
landmarks = PythonModelTracker.LandmarksGrabber.GetDefaultModelLandmarks(model3d,names_l)



renderer = ren.RendererOGLCudaExposed.get()

init_pos = model3d.default_state
value_range = model3d.high_bounds.data - model3d.low_bounds.data


dims = [52,53,55]
plt.figure(figsize=(15, 15))
steps = 12
for i in range(steps):
    # Setting param Vector
    init_pos[2] = 2700#10 * f loat(i)
    rot_q = at.quaternion_from_euler(0,0.9,0)
    init_pos[3] = rot_q[1]
    init_pos[4] = rot_q[2]
    init_pos[5] = rot_q[3]
    init_pos[6] = rot_q[0]
    # rot_q = at.quaternion_from_euler(0,3.14/2.,0)
    # init_pos[3] = rot_q[1]
    # init_pos[4] = rot_q[2]
    # init_pos[5] = rot_q[3]
    # init_pos[6] = rot_q[0]

    #init_pos[17] = model3d.low_bounds[17] + 0.1*f * (model3d.high_bounds[17] - model3d.low_bounds[17])

    for d in dims:
         init_pos[d] = model3d.low_bounds[d] + i * value_range[d][0] / float(steps)
    # init_pos[112] = model3d.default_state[112] * init_pos[110]
    # init_pos[113] = model3d.default_state[113] * init_pos[110]
    # init_pos[114] = model3d.default_sntate[114] * init_pos[110]
    # print init_pos[110]


    print('state:', init_pos)

    # Decoding
    decoding = decoder.quickDecode(init_pos)

    #calculating new landmark_positions.
    landmark_positions = landmarks_decoder.decode(decoding, landmarks)

    #Printing landmark info/new positions
    for l,p in zip(landmarks, landmark_positions):
        print('state:',model3d.default_state.data[0],model3d.default_state.data[1],model3d.default_state.data[2])
        print_lanmark_info(l)
        print('cur pos:', p.data[0],p.data[1],p.data[2])


    #Rendering model
    ru.render(renderer, mmanager, decoding, cam_meta, [1, 1],
              renderer.Culling.CullFront, model3d.n_bones)
    positions, normals, colors, issue, instance, vertex_id = ru.genMaps(renderer)
    normals_disp = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5

    #Projecting landmark positions to the image.
    new_positions_img = cam_meta.project(landmark_positions[0])
    for p in new_positions_img:
        cv2.rectangle(normals_disp,(int(p.x-2), int(p.y-2)),(int(p.x+2), int(p.y+2)),(0,0,1))

    #Visualizing
    plt.imshow(normals_disp)
    plt.waitforbuttonpress()


plt.close()


