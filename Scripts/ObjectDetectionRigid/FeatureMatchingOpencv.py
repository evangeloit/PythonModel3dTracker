import copy
import numpy as np

import cv2

import PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModelTracker.DatasetInfo as dsi
import PythonModelTracker.ModelTrackingResults as mtr
import PythonModelTracker.PyMBVAll as mbv


# Does not work!!
# Returns R,T for Y=TmatX --- X = TinvY
# def RigidInv(Tmat):
#     Tinit, S, Q = mbv.Core.DecomposeMatrix(Tmat)
#     Tinit = np.array(Tinit.__pythonize__()).T
#     R = at.quaternion_matrix([Q.w, Q.x, Q.y, Q.z]).T
#     T = - R[:3,:3].dot(Tinit)
#     Smat = np.eye(4,4)
#     Smat[0, 0] = 1 / S.x
#     Smat[1, 1] = 1 / S.y
#     Smat[2, 2] = 1 / S.z
#     Smat[3, 3] = 1
#     R[0:3,3] = T
#     print 'Rbef:\n',R
#     print 'Smat:\n', Smat
#     R = Smat.dot(R)
#     print 'Raft:\n',R
#     return R

def ApplyRigid(X,T):
    Y = T.dot(np.vstack([X, np.ones((1,X.shape[1]),X.dtype)]))
    return Y[:3,:]


def get_rigid(src, dst): # Assumes both or Nx3 matrices
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    # Compute covariance
    H = reduce(lambda s, (a,b) : s + np.outer(a, b), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T) # Rotation
    T = - R.dot(src_mean) + dst_mean # Translation
    return np.hstack((R, T[:, np.newaxis]))


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    print t

    return R, t

N = 10
f1 = 3
f2 = 4
# I1 2D object plane position
obj_x = (341, 448)
obj_w = obj_x[1] - obj_x[0]
obj_y = (138, 207)
obj_h = obj_y[1] - obj_y[0]
obj_corners = [[obj_x[0], obj_y[0]],
               [obj_x[1], obj_y[0]],
               [obj_x[1], obj_y[1]],
               [obj_x[0], obj_y[1]]]


params_ds = dsi.DatasetInfo()
params_ds.generate('box_01')
gt = mtr.ModelTrackingResults()
gt.load(params_ds.gt_filename)
mesh_manager = mbv.Core.MeshManager()
model3d = mbv.PF.Model3dMeta.create(str(Paths.model3d_dict[gt.models[0]]['path']))
decoder = model3d.createDecoder()
model3d.setupMeshManager(mesh_manager)
decoder.loadMeshTickets(mesh_manager)




gt1 = gt.get_model_states('box')[f1]
gt2 = gt.get_model_states('box')[f2]
print gt1,gt2
decoding = decoder.quickDecode(mbv.Core.DoubleVector(gt1))
Tmat = decoding[0].matrices[0]





grabber = AutoGrabber.create_di(params_ds)
grabber.seek(f1)
imgs, clbs = grabber.grab()
d1 = copy.deepcopy(imgs[0])
i1 = copy.deepcopy(imgs[1])
grabber.seek(f2)
imgs, _ = grabber.grab()
d2 = copy.deepcopy(imgs[0])
i2 = copy.deepcopy(imgs[1])
camera = clbs[0]
print camera

i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)


#cv2.waitKey(0)
mask = np.zeros_like(i1)
mask[138:207,341:448] = 255
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(i1,mask)
kp2, des2 = orb.detectAndCompute(i2,None)

print 'Found {0}, {1} features.'.format(len(kp1), len(kp2))

# BF Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

#FLANN matching
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.match(des1,des2)


# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
i3 = np.array([])
i3 = cv2.drawMatches(i1,kp1,i2,kp2,matches[:N],i3, flags=2)

# Get 3D Points from 2D feature locations.
p1_3d = np.zeros((N,3),dtype=np.float)
p2_3d = np.zeros((N,3),dtype=np.float)
for i,m in enumerate(matches[:N]):
    p2d = kp1[m.queryIdx].pt
    dval = d1[p2d[1], p2d[0]]
    p3d = camera.unproject(mbv.Core.Vector2(p2d[0], p2d[1]), np.float(dval))
    p1_3d[i][0] = p3d.x
    p1_3d[i][1] = p3d.y
    p1_3d[i][2] = p3d.z

    p2d = kp2[m.trainIdx].pt
    dval = d2[p2d[1], p2d[0]]
    p3d = camera.unproject(mbv.Core.Vector2(p2d[0], p2d[1]), np.float(dval))
    p2_3d[i][0] = p3d.x
    p2_3d[i][1] = p3d.y
    p2_3d[i][2] = p3d.z

#Get inverse transform

# Tinv = RigidInv(Tmat)
# print 'Tmat:', Tmat
# #print 'R, T:', R, T
# mp = ApplyRigid(p1_3d.T,Tinv)
# print mp.T

# store all the matches_good matches as per Lowe's ratio test.
matches_good = matches
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         matches_good.append(m)
print des1[:10,:]
if len(matches_good)>N:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_good]).reshape(-1, 1, 2)
    print src_pts
    M1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    print('Homography1:', M1)
    M2 = np.loadtxt('homography.txt')
    print('HomographySaved:', M2)
    h,w = i1.shape
    pts = np.float32(obj_corners).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M2)

    i2 = cv2.polylines(i2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

#affine = cv2.estimateAffine3D(np.array(p1_3d), np.array(p2_3d))
#print affine


cv2.imshow('i2', i2)
cv2.waitKey(0)

