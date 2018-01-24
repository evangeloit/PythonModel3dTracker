import PyMBVCore as core
import PyMBVAcquisition as acq
import PyMBVParticleFilter
import PyModel3dTracker as htpf
import PythonModelTracker.DatasetInfo as dsi
import PythonModelTracker.OpenPoseGrabber as opg
import PythonModelTracker.AutoGrabber as AutoGrabber


# Unit test.
if __name__ == '__main__':
    dataset = 'seq0'
    params_ds = dsi.DatasetInfo()
    params_ds.generate(dataset)
    grabber = AutoGrabber.create_di(params_ds)
    images, calibs = grabber.grab()
    print images[0].shape
    #op = opg.OpenPoseGrabber()
    #point_names, keypoints, clb = op.acquire(images, calibs)
    #op.ConvertIK(keypoints, clb)