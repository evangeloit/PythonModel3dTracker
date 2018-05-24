import numpy as np
import copy
import BlenderMBV.BlenderMBVLib.AngleTransformations as at
import PythonModel3dTracker.Paths as Paths
from PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools import Visualizer
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.ModelTrackingGui as GUI
import cv2
import BlenderMBV.BlenderMBVLib.BlenderMBVConversions as BMBVCONV

import PythonModel3dTracker.PythonModelTracker.Model3dUtils as M3DU


def ExtractGeom(bone_geometry, state):
    print state
    bone_vectors = bone_geometry.calcVectors(state, ['root', 'R.UArm', 'R.LArm'])
    bone_angles = bone_geometry.calcAngles(state, bone_vectors, [('root', 'R.UArm'), ('root', 'R.LArm'), ('R.UArm', 'R.LArm')])
    for b, v in bone_vectors.items(): print 'vec:', b, v
    for (b1, b2), a in bone_angles.items():
        print 'angle:', b1, b2, a


gui = ['opencv', 'blender'][1]
model_xml = Paths.model3d_dict['mh_body_male_custom_vector']['path']
model3d = mbv.PF.Model3dMeta.create(str(model_xml))
model_parts = model3d.parts

print('Loaded model from <', model_xml, '>', ', bones:', model3d.n_bones, ', dims:', model3d.n_dims)

mmanager = mbv.Core.MeshManager()
decoder = model3d.createDecoder()
model3d.setupMeshManager(mmanager )
decoder.loadMeshTickets(mmanager)

n_bones = model3d.n_bones
model_parts.genBonesMap()
print('Parts Map:', model_parts.parts_map)
print('Bones Map:', model_parts.bones_map)


# Setting camera/renderer
cam_meta = mbv.Lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
renderer = mbv.Ren.RendererOGLCudaExposed.get()

bone_geometry = M3DU.BoneGeometry(model3d, decoder)
state = model3d.default_state

if gui == 'opencv':
    value_range = model3d.high_bounds.data - model3d.low_bounds.data
    visualizer = Visualizer(model3d, mmanager, decoder, renderer)
    dims = [9,10]
    steps = 4
    for i in range(steps):
        # Setting param Vector
        state[2] = 2700#10 * f loat(i)
        rot_q = at.quaternion_from_euler(1.5,0+0.1*i,0)
        state[3] = rot_q[1]
        state[4] = rot_q[2]
        state[5] = rot_q[3]
        state[6] = rot_q[0]


        for d in dims:
            state[d] = model3d.low_bounds[d] + i * value_range[d][0] / float(steps)

        print 'state:', state

        ExtractGeom(bone_geometry, state)

        #Rendering model
        viz = visualizer.visualize(state,cam_meta)


        #Visualizing
        cv2.imshow("viz",viz)
        key = chr(cv2.waitKey(0) & 255)
        if key == 'q': break

elif gui == 'blender':
    continue_loop = True
    gui = GUI.ModelTrackingGuiZeromq()
    while continue_loop:
        gui_command = gui.recv_command()
        if gui_command.name == "quit":
            continue_loop = False

        if gui_command.name == "state":
            state_gui = gui.recv_state(model3d, state)
            if state_gui is not None: state = mbv.Core.DoubleVector(state_gui)
            ExtractGeom(bone_geometry, state)


        if gui_command.name == "init":
            gui.send_init(BMBVCONV.getFrameDataMBV(model3dmeta=model3d, state=state,
                                                 frames=[0, 1,100],
                                                 scale=0.001))


        if gui_command.name == "frame":
            f_gui = gui.recv_frame()
            if f_gui is not None:
                f = f_gui
                if (f > 100) or (f < 0): f = 0
                frame_data = BMBVCONV.getFrameDataMBV(model3dmeta=model3d, state=state,
                                                      mbv_camera=cam_meta,
                                                      frames=[0, f, 100],
                                                      images=None, scale=0.001)
                gui.send_frame(frame_data)
