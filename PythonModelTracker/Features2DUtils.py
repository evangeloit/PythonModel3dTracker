import numpy as np
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.DepthMapUtils as DMU
import PythonModel3dTracker.PythonModelTracker.GeomUtils as GU
from math import exp


def ApplyRigid(X,T):
    N = X.shape[1]
    dim = X.shape[0]
    Y = T.dot(np.vstack([X, np.ones((1,N),X.dtype)]))
    return Y[:dim,:]

def ApplyHomography(X,H):
    N = X.shape[1]
    dim = X.shape[0]
    Y = H.dot(np.vstack([X, np.ones((1,N),X.dtype)]))
    Y[0,:] /= Y[dim,:]
    Y[1,:] /= Y[dim, :]
    return Y[:dim,:]


def ConvertKeypointsArray(keypoints):
    # Get numpy array from opencv keypoints.
    p2d_np = np.zeros((2, len(keypoints)), dtype=np.float)
    for i,kp in enumerate(keypoints):
        p2d = mbv.Core.Vector2(kp.pt[0], kp.pt[1])
        p2d_np[0][i] = p2d.x
        p2d_np[1][i] = p2d.y
    return p2d_np


#keypoints: opencv feature detector points.
def GetPointsFromKeypoints(keypoints,camera,depth):
    # Get 3D Points from 2D feature locations.
    p3d_np = np.zeros((3,len(keypoints)),dtype=np.float)
    p2d_np = np.zeros((2, len(keypoints)), dtype=np.float)

    for i,kp in enumerate(keypoints):
        p2d = mbv.Core.Vector2(kp.pt[0], kp.pt[1])
        dval = DMU.GetMedianDepth(p2d,depth)#depth[p2d[1], p2d[0]]
        p3d = camera.unproject(p2d, np.float(dval))
        p2d_np[0][i] = p2d.x
        p2d_np[1][i] = p2d.y
        p3d_np[0][i] = p3d.x
        p3d_np[1][i] = p3d.y
        p3d_np[2][i] = p3d.z
    return p3d_np, p2d_np

# Generate landmarks for rigid objects.
# p3d_model: 3d points on the model space.
def CreateLandmarksFromPoints(model3d,p3d_model):
    landmarks_pos = mbv.Core.Vector3fStorage()
    landmark_names = mbv.Core.StringVector()
    geometry_names = mbv.Core.StringVector()
    bone_name = [b.key() for b in model3d.parts.bones_map][0]
    for i, pm in enumerate(p3d_model.T):
        landmark_names.append("{:05d}".format(i))
        geometry_names.append(bone_name)
        landmarks_pos.append(mbv.Core.Vector3(pm[0], pm[1], pm[2]))

    landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names,
                                                             geometry_names,
                                                             mbv.PF.ReferenceFrame.RFModel,
                                                             landmarks_pos,
                                                             model3d.parts.bones_map)
    return landmarks



def FilterKeypointsRandom(keypoints3d, keypoints2d, ratios=[0.1, 0.2]):
    #print ratios
    keypoints_out = mbv.Core.Vector3fStorage(keypoints3d)
    n = len(keypoints3d)
    ratio2d = min(ratios[0],ratios[1])
    ratio3d = max(ratios[0], ratios[1])
    xclude_indices_3d = np.unique(np.random.choice(n, int(ratio3d*n), replace=True))
    if xclude_indices_3d.size == 0:
        xclude_indices_2d = xclude_indices_3d
    else:
        xclude_indices_2d = np.unique(np.random.choice(xclude_indices_3d, int(ratio2d*n), replace=True))
    for xind in xclude_indices_3d:
        keypoints_out[xind].x = keypoints2d[xind].x
        keypoints_out[xind].y = keypoints2d[xind].y
        keypoints_out[xind].z = 0
    for xind in xclude_indices_2d:
        keypoints_out[xind].x = 0
        keypoints_out[xind].y = 0
        keypoints_out[xind].z = 0
    # print 'Keypoints_out:\n', keypoints_out
    return keypoints_out


def FilterKeypointsDepth(point_names, keypoints3d, keypoints2d, thres = 0.5):
    point_names_out = []
    keypoints3d_out = mbv.Core.Vector3fStorage()
    keypoints2d_out = mbv.Core.Vector2fStorage()
    acceptance_prob = []
    accepted_mask = []
    for names, p3ds, p2ds in zip( point_names, keypoints3d, keypoints2d):
        if len(p3ds) > 2:
            line_dist = GU.NormalizedLineDist(p3ds[0], p3ds[-1], p3ds[2:-1], 50)
        else:
            line_dist = 0
        cur_prob = exp(-(line_dist**2)/0.2)
        acceptance_prob.append(cur_prob)
        for n,p3d,p2d in zip(names, p3ds, p2ds):

            if cur_prob > thres:
                accepted_mask.append(True)
                point_names_out.append(n)
                keypoints3d_out.append(p3d)
                keypoints2d_out.append(p2d)
            else:
                accepted_mask.append(True)
                keypoints3d_out.append(mbv.Core.Vector3(p2d.x, p2d.y, 0))
                keypoints2d_out.append(p2d)

    return accepted_mask, point_names_out, keypoints3d_out, keypoints2d_out
