import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as DI
import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AG

dataset = DI.DatasetInfo()
dataset.load(Paths.datasets_dict['mhad_s01_a04'])

grabber = AG.create_di(dataset)

imgs,clbs = grabber.grab()


camera_meta = clbs[0]#mbv.Core.CameraMeta()

#Get the camera from camera_meta.
camera = camera_meta.camera



image_size_0 = (640, 480)
(fx,fy,cx,cy,zNear,zFar) = camera.getIntrinsics(image_size_0)
print image_size_0, 'fx:',fx,'fy:',fy, 'camera_center:',cx,cy, 'zNear:',zNear, 'zFar:',zFar

image_size_1 = (1000, 500)
camera.setIntrinsics(400,400,500,250,image_size_1[0],image_size_1[1],100,10000)
(fx,fy,cx,cy,zNear,zFar) = camera.getIntrinsics(image_size_1)
print image_size_1, 'fx:',fx,'fy:',fy, 'camera_center:',cx,cy, 'zNear:',zNear, 'zFar:',zFar
#(fx,fy,cx,cy,zNear,zFar) = camera.getIntrinsics(image_size_0)
#print image_size_0, 'fx:',fx,'fy:',fy, 'camera_center:',cx,cy, 'zNear:',zNear, 'zFar:',zFar

#set the camera to camera_meta
camera_meta.camera = camera