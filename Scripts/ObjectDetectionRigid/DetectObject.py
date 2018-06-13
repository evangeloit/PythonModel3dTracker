import copy
import numpy as np
import os
import pickle

import BlenderMBVLib.AngleTransformations as at
import BlenderMBVLib.RenderingUtils as ru
import PyModel3dTracker as pm3d
import cv2

import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.Features2DUtils as f2d
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv

did = 'box_03'
appearance_filename = os.path.join(Paths.objdetect, 'box_appearance.pck')
f = 2
N=10


params_ds = dsi.DatasetInfo()
params_ds.generate(did)
gt = mtr.ModelTrackingResults()
gt.load(params_ds.gt_filename)
model_name = gt.models[0]
mesh_manager = mbv.Core.MeshManager()
model3d = mbv.PF.Model3dMeta.create(str(Paths.model3d_dict[model_name]['path']))
model3d.parts.genBonesMap()
decoder = model3d.createDecoder()
model3d.setupMeshManager(mesh_manager)
decoder.loadMeshTickets(mesh_manager)
renderer = mbv.Ren.RendererOGLCudaExposed.get()

grabber = AutoGrabber.create('oni', [''])
#grabber = AutoGrabber.create_di(params_ds)
#grabber.seek(f)
for f in range(20):
    imgs, clbs = grabber.grab()
    depth = copy.deepcopy(imgs[0])
    rgb = copy.deepcopy(imgs[1])
    camera = clbs[0]
    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    print camera


    (p3d_model, p3d_def_np, p2d1, des1) = pickle.load(open(appearance_filename,'rb'))
    #p3d_model = np.loadtxt('box_p3dmodel.txt')
    #p2d1 = np.loadtxt('box_p2d.txt')
    #des1 = np.loadtxt('box_desc.txt',dtype=np.uint8)

    orb = cv2.ORB_create()
    kp2, des2 = orb.detectAndCompute(img,np.ones_like(img))
    p3d_def = mbv.Core.Vector3fStorage(p3d_def_np.T)


    # BF Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    # FLANN matching
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.match(des1,des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches_good = matches

    if len(matches_good)>N:
        src_pts = np.float32([(p2d1[:,m.queryIdx][0],p2d1[:,m.queryIdx][1]) for m in matches_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_good]).reshape(-1, 1, 2)


        src_pts_vec = mbv.Core.Vector3fStorage([p3d_def[m.queryIdx] for m in matches_good])
        dst_pts_vec = mbv.Core.Vector2fStorage(
            [mbv.Core.Vector2([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]) for m in matches_good])
        print dst_pts, dst_pts_vec
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        Tposest = mbv.Core.DoubleVector()
        pm3d.posest(Tposest, dst_pts_vec, src_pts_vec, 0.5, camera)
        matchesMask = mask.ravel().tolist()
        print('Homography:',M)
        print('RigidBody:',Tposest)

        p2d_hom = f2d.ApplyHomography(p2d1, M)
        ru.disp_points_np(p2d_hom,rgb)



    state = model3d.default_state
    R,_ = cv2.Rodrigues(np.array([Tposest[0], Tposest[1], Tposest[2]]))

    # Rmat = np.eye(4,4,dtype=R.dtype)
    # Rmat[0:3][0:3] = R
    quat = at.quaternion_from_matrix(R)
    print 'Rmat',R
    state[0] = Tposest[3]
    state[1] = Tposest[4]
    state[2] = Tposest[5]
    state[3] = quat[1]
    state[4] = quat[2]
    state[5] = quat[3]
    state[6] = quat[0]
    state[7] = 300
    state[8] = 90
    state[9] = 20
    print 'State:', state
    viz = ru.visualize_overlay(renderer,mesh_manager,decoder,state,camera,rgb,model3d.n_bones)
    cv2.imshow("viz",viz)
    #cv2.imshow("rgb",rgb)
    # cv2.imshow("i2",i2)
    # cv2.imshow('i3', i3)
    # cv2.imshow('mask', mask)
    cv2.waitKey(0)