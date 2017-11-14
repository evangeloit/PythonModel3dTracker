import numpy as np
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv


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

#keypoints: opencv feature detector points.
def GetPointsFromKeypoints(keypoints,camera,depth):
    # Get 3D Points from 2D feature locations.
    p3d_np = np.zeros((3,len(keypoints)),dtype=np.float)
    p2d_np = np.zeros((2, len(keypoints)), dtype=np.float)

    for i,kp in enumerate(keypoints):
        p2d = kp.pt
        dval = depth[p2d[1], p2d[0]]
        p3d = camera.unproject(mbv.Core.Vector2(p2d[0], p2d[1]), np.float(dval))
        p2d_np[0][i] = p2d[0]
        p2d_np[1][i] = p2d[1]
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