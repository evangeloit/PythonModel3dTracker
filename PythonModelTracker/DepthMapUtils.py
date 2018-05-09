import numpy as np
import PythonModel3dTracker.PyMBVAll as mbv



def Filter3DRect(depthmap,bb,z,depth_cutoff):
    depthmap[:, 0:bb[0]] = 0
    depthmap[0:bb[1], :] = 0
    depthmap[:, bb[0] + bb[2]:] = 0
    depthmap[bb[1] + bb[3]:, :] = 0
    depthmap[depthmap > z + depth_cutoff] = 0
    depthmap[depthmap < z - depth_cutoff] = 0
    return depthmap

def GetMedianDepths(points, depthmap, w=4):
    depths = []
    for p in points:
        depths.append(GetMedianDepth(p,depthmap,w))
    return depths

def GetMedianDepth(p,depthmap,w=4):
    x = p.x
    y = p.y
    height = depthmap.shape[0]
    width = depthmap.shape[1]
    if (x > 0) and (y > 0) and (x < width) and (y < height):
        d = np.median(depthmap[max(y-w, 0):min(y+w, height-1),
                               max(x-w, 0):min(x+w, width-1)])
        #d = np.nan_to_num(d)
    else:
        d = 0
    return d

def UnprojectPoints(points2d, camera, depth, w=4):
    kp_depths = mbv.Core.SingleVector(GetMedianDepths(points2d, depth, w))
    points3d = camera.unproject(points2d, kp_depths)
    return points3d

def UnprojectPointSets(points2dsets, camera, depth, w=4):
    keypoints3dsets = []
    for kps2d in points2dsets:
        keypoints3dsets.append(UnprojectPoints(kps2d,camera,depth,w))
    return keypoints3dsets