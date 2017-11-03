import PythonModelTracker.PyMBVAll as mbv
import PyModel3dTracker as pm3d

# cf = mbv.Core.CameraFrustum()
# cf.setIntrinsics(610,610,320,240,640,480,100,10000)
# camera = mbv.Core.CameraMeta()
# camera.width = 640
# camera.height = 480
# camera.camera = cf
camera = mbv.Lib.RGBDAcquisitionSimulation.getDefaultCalibration()

p3d = mbv.Core.Vector3fStorage()
p3d.append(mbv.Core.Vector3([0,0,100]))
p3d.append(mbv.Core.Vector3([0,100,100]))
p3d.append(mbv.Core.Vector3([100,100,100]))
p3d.append(mbv.Core.Vector3([100,0,100]))
# p3d.append(mbv.Core.Vector3([4,4,120]))
# p3d.append(mbv.Core.Vector3([2,2,110]))
# p3d.append(mbv.Core.Vector3([2,2,130]))
# p3d.append(mbv.Core.Vector3([2,2,140]))
# p3d.append(mbv.Core.Vector3([2,2,150]))
# p3d.append(mbv.Core.Vector3([12,12,160]))
# p3d.append(mbv.Core.Vector3([21,21,170]))
# p3d.append(mbv.Core.Vector3([20,20,180]))
p2d = camera.project(p3d)
print p2d

for p in p3d:
    p.z += 50

print p3d

pose = mbv.Core.DoubleVector()

pm3d.posest(pose,p2d,p3d,0.9,camera)

print 'pose:',pose