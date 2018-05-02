import copy
import os
import pickle

import BlenderMBV.BlenderMBVLib.RenderingUtils as ru

import PyModel3dTracker as pm3d
import cv2

from PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools import Visualizer
from PythonModel3dTracker.PythonModelTracker.DatasetInfo import DatasetInfo
import PythonModel3dTracker.PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.Paths as Paths
from ObjectDetection.RigidObjectOptimizer import RigidObjectOptimizer

if __name__ == '__main__':
    mesh_manager = mbv.Core.MeshManager()

    # Loading the model.
    model_name = 'mh_body_male_custom'
    model_path = str(Paths.model3d_dict[model_name]['path'])
    model3d = mbv.PF.Model3dMeta.create(model_path)
    model3d.parts.genBonesMap()
    decoder = model3d.createDecoder()
    model3d.setupMeshManager(mesh_manager)
    decoder.loadMeshTickets(mesh_manager)

    # Loading the dataset.
    dataset_filename = '/home/mad/Development/Datasets/human_tracking/co4robots/ms1_gestures/ms1_gestures_00.json'
    dataset_info = DatasetInfo()
    dataset_info.load(dataset_filename)
    grabber = AutoGrabber.create_di(dataset_info)

    # Initializing results for logging the states.
    #results_filename = Paths.datasets + '/object_tracking/co4robots/{}_res.json'.format(did)
    #results = mtr.ModelTrackingResults(did)


    # Initializing the renderer.
    renderer = pm3d.Model3dObjectiveFrameworkRendering. \
                generateDefaultRenderer(2048, 2048, "opengl",
                                        model3d.n_bones,
                                        mbv.Ren.RendererOGLBase.Culling.CullNone)


    # Initializing the optimizer.
    optimizer_settings = {
        "particles": 64,
        "generations": 128,
        "depth_cutoff": 1500,
        "variances": [150, 150, 150, 0.2, 0.2, 0.2, 0.2] + [0.4]*19,
        "tile_size": (128, 128)
    }
    object_optimizer = RigidObjectOptimizer(mesh_manager, renderer, decoder, model3d, optimizer_settings)

    # Initializing the visualizer.
    visualizer = Visualizer(model3d, mesh_manager, decoder, renderer)


    for f in range(5):
        # Acquiring input frames and camera calibration.
        imgs, clbs = grabber.grab()
        depth = copy.deepcopy(imgs[0])
        rgb = copy.deepcopy(imgs[1])
        camera = clbs[0]
        camera_frust = camera.camera
        # Manually set the intrinsics.
        #camera_frust.setIntrinsics(531.5, 532.4, 314.6, 252.5, camera.width, camera.height, camera_frust.zNear, camera_frust.zFar)
        #camera.camera = camera_frust
        print 'Camera calibration:', camera


        #Setting the initial state here.n
        state_init = mbv.Core.DoubleVector([-150.0, -30.0, 2300.0, 0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print "Initial State Vector:", state_init

        #Optimizing.
        state_opt = object_optimizer.optimize(imgs,clbs,state_init)

        # Logging and visualizing the results.
        print "Optimized State Vector:", state_opt
        # results.add(f,model_name,state_opt)
        viz_init = visualizer.visualize_overlay(state_init, camera, rgb)
        viz_opt = visualizer.visualize_overlay(state_opt, camera, rgb)
        cv2.imshow("viz_init", viz_init)
        cv2.imshow("viz_opt", viz_opt)
        cv2.imshow("depth", depth)
        cv2.waitKey(0)

    # Saving state vectors log.
    #results.save(results_filename)