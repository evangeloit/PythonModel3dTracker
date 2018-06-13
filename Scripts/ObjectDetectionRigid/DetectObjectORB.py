import copy
import os
import pickle

import BlenderMBVLib.RenderingUtils as ru
import PyModel3dTracker as pm3d
import cv2

import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv
from ObjectDetection.RigidObjectDetectorORB import RigidObjectDetectorORB, ObjectData
from ObjectDetection.RigidObjectOptimizer import RigidObjectOptimizer

if __name__ == '__main__':

    model_name = 'box'
    did = 'box_eps_02'
    dataset_filename = Paths.datasets + '/object_tracking/co4robots/{}.oni'.format(did)
    results_filename = Paths.datasets + '/object_tracking/co4robots/{}_res.json'.format(did)
    appearance_filename = os.path.join(Paths.objdetect, 'box_eps_appearance.pck')
    min_matches = 10
    results = mtr.ModelTrackingResults(did)


    # did = 'box_03'
    # f = 2
    # params_ds = dsi.DatasetInfo()
    # params_ds.generate(did)
    # gt = mtr.ModelTrackingResults()
    # gt.load(params_ds.gt_filename)
    # model_name = gt.models[0]
    grabber = AutoGrabber.create('oni', [dataset_filename])
    mesh_manager = mbv.Core.MeshManager()
    model3d = mbv.PF.Model3dMeta.create(str(Paths.model3d_dict[model_name]['path']))
    model3d.parts.genBonesMap()
    decoder = model3d.createDecoder()
    model3d.setupMeshManager(mesh_manager)
    decoder.loadMeshTickets(mesh_manager)
    renderer = pm3d.Model3dObjectiveFrameworkRendering. \
                generateDefaultRenderer(2048, 2048, "opengl",
                                        model3d.n_bones,
                                        mbv.Ren.RendererOGLBase.Culling.CullNone)
    object_appearance = pickle.load(open(appearance_filename, 'rb'))
    object_detector = RigidObjectDetectorORB([ObjectData(model3d, object_appearance)], min_matches)
    object_optimizer = RigidObjectOptimizer(mesh_manager, renderer, decoder, model3d)


    for f in range(20):
        imgs, clbs = grabber.grab()
        depth = copy.deepcopy(imgs[0])
        rgb = copy.deepcopy(imgs[1])
        camera = clbs[0]
        camera_frust = camera.camera
        camera_frust.setIntrinsics(531.5, 532.4, 314.6, 252.5,
                                   camera.width, camera.height, camera_frust.zNear, camera_frust.zFar)
        #camera_frust.zNear = 360
        camera.camera = camera_frust
        print clbs[0]

        state_det = object_detector.detect(imgs, clbs)[0]
        if state_det is not None:
            state_opt = object_optimizer.optimize(imgs,clbs,state_det)
            visible_depth, visible_points = object_optimizer.extract_visible(imgs,clbs,state_opt,30)
            print 'visible points number:', len(visible_points)

        if state_det is None:
            viz_det = rgb
            viz_opt = rgb
            visible_depth = depth

        else:
            viz_det = ru.visualize_overlay(renderer,mesh_manager,decoder,state_det,camera,rgb,model3d.n_bones)
            viz_opt = ru.visualize_overlay(renderer, mesh_manager, decoder, state_opt, camera, rgb, model3d.n_bones)
            results.add(f,model_name,state_opt)

        cv2.imshow("viz_det",viz_det)
        cv2.imshow("viz_opt", viz_opt)
        cv2.imshow("visible_depth", visible_depth)
        cv2.waitKey(1)
    results.save(results_filename)