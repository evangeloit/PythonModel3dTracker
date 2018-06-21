import numpy as np

def MeanLineDist(p1, p2, points):
    p1 = np.array([p1.x, p1.y, p1.z])
    p2 = np.array([p2.x, p2.y, p2.z])
    dists = []
    for p in points:
        p = np.array([p.x, p.y, p.z])
        cur_dist = np.linalg.norm(np.cross(p2-p1, p-p1))/np.linalg.norm(p2-p1)
        #print p1, p2, p, cur_dist
        dists.append(cur_dist)
    mean_dist = np.average(dists)
    return mean_dist


def NormalizedLineDist(p1,p2,points,max_dist=50):
    #print 'p1p2points', p1, p2, points
    mean_dist = MeanLineDist(p1,p2,points)
    norm_dist = min(1, mean_dist/max_dist)
    return norm_dist