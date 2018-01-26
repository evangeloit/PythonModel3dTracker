import copy
import numpy as np
import os
import pickle

import BlenderMBV.BlenderMBVLib.AngleTransformations as at
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru
import cv2

import PythonModel3dTracker.PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.Features2DUtils as f2d
import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv
from PythonModel3dTracker.ObjectDetection.RigidObjectDetectorORB import ObjectAppearance
import PythonModel3dTracker.Paths as Paths

# Returns R,T for Y=TmatX --- X = TinvY
# def RigidInv(Tin):
#
#     Tvec, S, Q = mbv.Core.DecomposeMatrix(Tin)
#     T = np.zeros([3,1])
#     T[0] = Tvec.x
#     T[1] = Tvec.y
#     T[2] = Tvec.z
#     print 'T:\n',T
#     R = at.quaternion_matrix([Q.w, Q.x, Q.y, Q.z])
#     print 'R:\n', R
#     T = -R[:3,:3].dot(T)
#     print '-Rt:\n',T
#     R[0:3,3] = T.T
#     print 'Tout:\n',R
#
#     return R




did = 'box_eps_01'
appearance_filename = os.path.join(Paths.objdetect, 'box_eps_appearance_v01.pck')
frames = [0,1,2,3,4,5,6]


# Initializing MBV rendering stack.
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
grabber = AutoGrabber.create_di(params_ds)
orb = cv2.ORB_create()
p3d_ren_all = None


# Getting Frame
f = frames[0]
for f in frames:
    grabber.seek(f)
    imgs, clbs = grabber.grab()
    depth = copy.deepcopy(imgs[0])
    rgb = copy.deepcopy(imgs[1])
    camera = clbs[0]
    print 'Camera Intrinsics:', camera.camera.getIntrinsics(camera.size)
    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gt_state = mbv.Core.DoubleVector(gt.get_model_states('box')[f])
    def_state = mbv.Core.DoubleVector([0, 0, 0, 0., 0., 0., 1., gt_state[7], gt_state[8], gt_state[9]])
    dimensions = [gt_state[7], gt_state[8], gt_state[9]]
    print 'Default Pose State:',gt_state

    # Extracting features from the model area.
    decoding = decoder.quickDecode(gt_state)
    ru.render(renderer,mesh_manager,decoding,camera,[1,1],mbv.Ren.RendererOGLBase.Culling.CullNone,1)
    positions, normals, colors, issue, instance, V = ru.genMaps(renderer)
    mask = (issue > 0).astype(img.dtype)
    kernel = np.ones((5,5),mask.dtype)
    mask = cv2.erode(mask,kernel,iterations = 1)
    kp, des = orb.detectAndCompute(img,mask)

    # Transforming points to model space.
    Tmat = decoding[0].matrices[0]
    Tinit = np.array(Tmat.__pythonize__()).T
    Tinv = np.linalg.inv(Tinit)
    p3d_dpt,p2d = f2d.GetPointsFromKeypoints(kp,camera,depth)

    p3d_ren = np.zeros_like(p3d_dpt)
    for i,(p, p_ren, p_dpt) in enumerate(zip(p2d.T, p3d_ren.T ,p3d_dpt.T )):
        p_ren = positions[p[1],p[0],:]
        p3d_ren[:,i] = p_ren
        print i,p, p_ren, p_dpt, np.linalg.norm(p_ren - p_dpt)

    p3d_model = f2d.ApplyRigid(p3d_ren,Tinv)
    print 'Tinit:', '\n' ,Tinit
    print 'Tinv:', '\n' ,Tinv


    # Generating Landmarks & transforming points in default (neutral) pose.
    landmarks = f2d.CreateLandmarksFromPoints(model3d, p3d_model)
    landmarks_decoder = mbv.PF.LandmarksDecoder(decoder)
    p3d_def = landmarks_decoder.decode(def_state,landmarks)
    p3d_defpose = np.array(p3d_def.__pythonize__()).T
    print 'Default Pose State:',def_state
    for i,l in enumerate(landmarks):
        print l.name, l.linked_geometry, l.pos, p3d_def[i]



    #Visualizing
    cf = camera.camera
    cf.position = mbv.Core.Vector3([0,0,-800])
    q = at.quaternion_from_euler(0, 0, 0)
    cf.orientation = mbv.Core.Quaternion([q[1],q[2],q[3],q[0]])
    camera.camera = cf
    viz = ru.visualize(renderer,mesh_manager,decoder,def_state,camera,model3d.n_bones,[p3d_def])
    ru.disp_points_np(p2d,rgb)
    cv2.imshow('img', img * mask)
    cv2.imshow("viz",viz)
    cv2.imshow("rgb",rgb)
    cv2.waitKey(0)

    if p3d_ren_all is None:
        p3d_ren_all = p3d_ren
        p3d_model_all = p3d_model
        p3d_defpose_all = p3d_defpose
        p2d_all = p2d
        des_all = des
    else:
        p3d_ren_all = np.hstack((p3d_ren_all, p3d_ren))
        p3d_model_all = np.hstack((p3d_model_all, p3d_model))
        p3d_defpose_all =  np.hstack((p3d_defpose_all, p3d_defpose))
        p2d_all = np.hstack((p2d_all, p2d))
        des_all = np.vstack((des_all, des))
    print(p2d_all.shape, des_all.shape)

# Save appearance
object_appearance = ObjectAppearance(dimensions, p3d_model_all, p3d_defpose_all, p2d_all, des_all)
pickle.dump(object_appearance,open(appearance_filename,'wb'))


