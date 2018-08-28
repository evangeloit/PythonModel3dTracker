import cv2

#import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.Landmarks.OpenPoseGrabber as opg
import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.Paths as Paths
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru


# Unit test.
if __name__ == '__main__':
    dataset = '/home/evangeloit/Desktop/alex_far_01.oni'
    # dataset = 'mhad_s01_a04'
    params_ds = dsi.DatasetInfo()
    params_ds.generate(dataset)
    grabber = AutoGrabber.create_di(params_ds)
    images, calibs = grabber.grab()

    # op = opg.OpenPoseGrabber(model_op_path=Paths.models_openpose)
    op = opg.OpenPoseGrabber(model_op_path=Paths.models_openpose, net_size=(160, 128), res_size=(320, 240))
    point_names, keypoints, keypoints2d, clb, _ = op.acquire(images, calibs)
    viz = images[1]
    viz = ru.disp_points(keypoints2d[0], viz)

    cv2.imshow("viz",viz)
    cv2.waitKey(0)