import PyMBVCore as core
import numpy as np
import cv2
import math



class BulletDebugDraw:
    def __init__(self):
        self.points3da = np.zeros([0,3])
        self.points3db = np.zeros([0,3])

    def drawLine(self,p1,p2,c1,c2):
        self.points3da = np.append(self.points3da, p1.data.transpose(), axis=0)
        self.points3db = np.append(self.points3db, p2.data.transpose(), axis=0)
        return True
    def drawSphere(self,radius,pos,q,color): return False
    def drawTriangle(self,v0,v1,v2,n0,n1,n2,color): return False

    def drawContactPoint(self,point,normal,distance,lifeTime,color): return False

    def drawAabb(self,p1,p2,color): return False

    def drawTransform(self,pos,q,length): return False

    def drawArc(self,  center,  normal,  axis,
                  radiusA,  radiusB,  minAngle,  maxAngle,
                   color,  drawSect,  stepDeg): return False

    def drawSpherePatch(self,center, up, axis,radius,minTh,maxTh, minPs, maxPs, color, stepDeg, drawCenter): return False
    def drawBox(self,bbMin,bbMax,pos,rot,color): return False
    def drawCapsule(self,radius,halfHeight,upAxis,pos,q,color): return False
    def drawCylinder(self,radius,halfHeight,upAxis,pos,q,color): return False
    def drawCone(self,radius,height,upAxis,pos,q,color): return False
    def drawPlane(self,normal,offset,pos,q,color): return False


def ProjectDrawLinesOpencv(points3da_arr,points3db_arr,cam_meta,img):
    points3da = core.Vector3fStorage()
    points3db = core.Vector3fStorage()
    for p1,p2 in zip(points3da_arr,points3db_arr):
        points3da.append(core.Vector3(p1[0],p1[1],p1[2]))
        points3db.append(core.Vector3(p2[0],p2[1],p2[2]))
    points2da = cam_meta.project(points3da)
    points2db = cam_meta.project(points3db)

    points2da_arr = np.zeros([0,2])
    points2db_arr = np.zeros([0,2])
    for p1,p2 in zip(points2da,points2db):
        points2da_arr = np.append(points2da_arr, p1.data.T, axis=0)
        points2db_arr = np.append(points2db_arr, p2.data.T, axis=0)

    # for p3d,p2d in zip(bdd.points3da,points2da_arr):
    #     print '{0} -> {1}'.format(p3d,p2d)
    #print(np.min(points2da_arr,axis=0), np.max(points2da_arr,axis=0))

    for p1,p2 in zip(points2da_arr,points2db_arr):
        if math.isnan(p1[0]) or math.isinf(p1[0]) or\
           math.isnan(p1[1]) or math.isinf(p1[1]) or\
           math.isnan(p2[0]) or math.isinf(p2[0]) or\
           math.isnan(p2[1]) or math.isinf(p2[1]):
            pass
        else:
            cv2.line(img,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,255,255))


# Create Compound Shape
# compound_cyl = phys.CompoundShape()
# compound_cyl.scale= core.Vector3(1,1,1)
# euler_angles = [0,0,0]
# rot_q = at.quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2])
# print rot_q, core.Quaternion(rot_q[0],rot_q[1],rot_q[2],rot_q[3]).data
# compound_cyl.addShape(core.Vector3(0,0,0),core.Quaternion(rot_q[0],rot_q[1],rot_q[2],rot_q[3]),cyl_shape)