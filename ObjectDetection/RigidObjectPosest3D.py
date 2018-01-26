import numpy as np
import BlenderMBV.BlenderMBVLib.AngleTransformations as at

default_ransac_settings = {
    "Nreps": 5000,
    "dmax": 30,
    "Npoints":3
}


# GET Rigid Transform p->q.
def CalcRigidTransformRansac(p, q, settings=default_ransac_settings):
    N = settings["Nreps"]
    dmax = settings["dmax"]
    M = settings["Npoints"]
    Np = np.shape(p)[1]
    #print "q:\n", q
    inliers_ratio = 0
    for i in range(N):
        sel_points_idx = np.random.choice(Np, M, replace=False)
        #print "sel_points_idx",sel_points_idx
        Rt = CalcRigidTransformSVD(p[:, sel_points_idx], q[:, sel_points_idx])
        cur_inliers_idx, cur_outliers_idx, cur_ratio = CalcInliersRatio(Rt, p, q, dmax)
        if cur_ratio >= inliers_ratio:
            inliers_ratio = cur_ratio
            inliers_idx = cur_inliers_idx
            outliers_idx = cur_outliers_idx
    #print "inliers_ratio:", inliers_ratio
    Rt = CalcRigidTransformSVD(p[:, inliers_idx], q[:, inliers_idx])
    #print "Ransac Rt:\n", Rt
    return Rt, outliers_idx


def toHom(p):
    n = np.shape(p)[1]
    phom = np.append(p, np.ones((1, n)), axis=0)
    return phom

def toCart(phom):
    d = np.shape(phom)[0] - 1
    p = phom[0:d, :]
    for i in range(d): p[0, :] /= phom[d, :]
    return p


def CalcInliersRatio(Rt, p, q, dmax):
    n = np.shape(p)[1]
    phom = toHom(p)
    p_hom = np.dot(Rt, phom)
    p_ = toCart(p_hom)
    eucl_dist = np.linalg.norm(q - p_, axis=0)
    inliers_mask = (eucl_dist <= dmax)
    cur_inliers_idx = np.where(inliers_mask)[0]
    outliers_mask = (eucl_dist > dmax)
    cur_outliers_idx = np.where(outliers_mask)[0]
    ratio = np.count_nonzero(inliers_mask) / float(n)
    #print "eucl_dist:", eucl_dist
    #print "inliers_ratio:", ratio
    return cur_inliers_idx, cur_outliers_idx, ratio


# GET Rigid Transform p->q.
def CalcRigidTransformSVD(p, q):
    d = np.shape(q)[0]
    n = np.shape(q)[1]
    qm = (np.mean(q, axis=1)[np.newaxis]).T
    pm = (np.mean(p, axis=1)[np.newaxis]).T
    Y = q - np.tile(qm, (1, n))
    X = p - np.tile(pm, (1, n))
    S = np.dot(X, Y.T)
    [U, _, V] = np.linalg.svd(S)
    UT = U.T
    VT = V.T
    VU = np.dot(VT, UT)
    #print "VU\n", VU
    detVU = np.linalg.det(VU)
    #print "detVU:", detVU
    I = np.eye(d)
    I[d - 1, d - 1] = detVU
    #print "I\n",I
    R = np.dot(np.dot(VT, I), UT)
    t = qm - np.dot(R, pm)
    Rt = at.compose_matrix(angles=at.euler_from_matrix(R), translate=t.flatten())
    #print 'p:\n', p
    #print 'q:\n', q
    #print 'SVD Rt:\n', Rt
    return Rt