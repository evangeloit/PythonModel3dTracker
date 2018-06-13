import os

import cv2

import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.Paths as Paths

grabber = AutoGrabber.create('oni', [''])


did = 'objectiw3_00'
rel_path = 'object_tracking/co4robots/'
output_dir = os.path.join(Paths.datasets, rel_path, did)
gt_filename = did + '_gt.json'
ds_filename = did + '.json'
gt_path = os.path.join(Paths.datasets, rel_path, gt_filename)
ds_path = os.path.join(Paths.datasets, rel_path, ds_filename)

model_name = 'box'
model3d = mbv.PF.Model3dMeta.create(str(Paths.model3d_dict[model_name]['path']))
state = model3d.default_state
gt = mtr.ModelTrackingResults()
gt.did = did


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Main loop
f = 0
continue_loop = True
while continue_loop:
    images, calibs = grabber.grab()
    depth = images[0]
    rgb = images[1]

    rgb_fname = os.path.join(output_dir,"rgb_{:04d}.png".format(f))
    dpt_fname = os.path.join(output_dir, "dpt_{:04d}.png".format(f))
    print 'saving ', rgb_fname, dpt_fname
    cv2.imwrite(rgb_fname,rgb)
    cv2.imwrite(dpt_fname,depth)
    gt.add(f,model_name,state)
    f += 1
    cv2.imshow('rgb', rgb)
    cv2.imshow('depth', depth)
    key = chr(cv2.waitKey(33) & 255)
    if key == 'q': continue_loop = False


if f > 0:
    params_ds = dsi.DatasetInfo()
    params_ds.generate(output_dir)
    params_ds.gt_filename = gt_path
    gt.save(gt_path)
    params_ds.save(ds_path)
    Paths.datasets_dict[did] = ds_path
    Paths.save_datasets_dict(Paths.datasets_dict)


