import PyMBVRendering as ren
import numpy as np

import BlenderMBV.BlenderMBVLib.AngleTransformations as at
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru
import PyMBVCore as core
import PyMBVLibraries as lib
import PyMBVParticleFilter as mpf
import cv2

import PythonModel3dTracker.PythonModelTracker.Landmarks.LandmarksGrabber as ldm

model3d_xml = Paths.model3d_dict['mh_body_male_custom']['path']

model3d = mpf.Model3dMeta.create(str(model3d_xml))

mmanager = core.MeshManager()
model3d.setupMeshManager(mmanager)
model3d.parts.genBonesMap()
decoder = model3d.createDecoder()
decoder.loadMeshTickets(mmanager)
renderer = ren.RendererOGLCudaExposed.get()

camera = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
print camera, model3d.default_state

bone_names = model3d.parts.parts_map['all']
landmarks = ldm.GetDefaultModelLandmarks(model3d, bone_names)
landmarks_decoder = mpf.LandmarksDecoder()




state = model3d.default_state
rot_q = at.quaternion_from_euler(1.5,0,0)
state[3] = rot_q[1]
state[4] = rot_q[2]
state[5] = rot_q[3]
state[6] = rot_q[0]

decoding = decoder.quickDecode(state)
landmark_positions = landmarks_decoder.decode(decoding, landmarks)
#img = ru.visualize(renderer,mmanager,decoder,state,camera,model3d.n_bones,landmark_positions)
ru.render(renderer, mmanager, decoding, camera, [1, 1], ren.RendererOGLBase.Culling.CullFront, model3d.n_bones)
positions, normals, colors, issue, instance, V = ru.genMaps(renderer)

landmarks_positions2d = camera.project(landmark_positions[0])

for n,l2d, l3d in zip(bone_names,landmarks_positions2d, landmark_positions[0]):
    pl = np.array([l3d.x,l3d.y,l3d.z])
    ps = positions[l2d.y,l2d.x,:]
    ps_c = camera.unproject(core.Vector2(l2d.x, l2d.y), l3d.z)
    dist = np.linalg.norm(pl-ps)
    dist_c = np.linalg.norm(pl - np.array([ps_c.x, ps_c.y, ps_c.z]))
    print n,'2d({0},{1}), 3d({2}), surf({3}), cor({3}), dist:{4},{5}'.format(l2d.x,l2d.y,pl,ps,dist,dist_c )

# Displaying the normals.
viz = 255*(positions * 0.5 + 0.5)
viz[issue == 0] = 0
viz = viz.astype(np.uint8)
viz = ru.visualize_points(viz, camera, landmark_positions)


cv2.imshow('model', viz)
cv2.waitKey(0)
