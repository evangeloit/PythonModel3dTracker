import os

import cv2

import PythonModel3dTracker.PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.ModelTrackingGui as mtg
import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.Paths as Paths

visualize = {'enable':True,
             'labels':True, 'depth':True, 'rgb':True, 'wait_time':33}

grabber = AutoGrabber.create('oni',[''])

gui = mtg.ModelTrackingGuiOpencv(visualize=visualize, init_frame=0)

did = 'boxiw3_large'
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



# Main loop
f = 0
f_counter = 0
continue_loop = True
while continue_loop:
    gui_command = gui.recv_command()
    if gui_command.name == "quit":
        continue_loop = False

    if gui_command.name == "init":
        gui_command.name = "frame"
        gui.next_frame = f

    if gui_command.name == "frame":
        f_gui = gui.recv_frame()
        if f_gui is not None:
            f = f_gui
            # print 'frame  :', f
            images, calibs = grabber.grab()
            depth = images[0]
            rgb = images[1]
            frame_data = mtg.FrameDataOpencv(depth, None, rgb, f )
            gui.send_frame(frame_data)

    if gui_command.name == "save":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        rgb_fname = os.path.join(output_dir,"rgb_{:04d}.png".format(f_counter))
        dpt_fname = os.path.join(output_dir, "dpt_{:04d}.png".format(f_counter))
        print 'saving ', rgb_fname, dpt_fname
        cv2.imwrite(rgb_fname,rgb)
        cv2.imwrite(dpt_fname,depth)
        gt.add(f_counter,model_name,state)
        f_counter += 1

# params_ds = dsi.DatasetInfo(did=did,format='SFImage',stream_filename=[did+"/dpt_%04d.png", did+"/rgb_%04d.png"],
#                             calib_filename='calib.txt',flip_images=None,
#                             limits=[0,f_counter-1],gt_filename=gt_filename,landmarks=None,background=None)
if f_counter > 0:
    params_ds = dsi.DatasetInfo()
    params_ds.generate(output_dir)
    params_ds.gt_filename = gt_path
    gt.save(gt_path)
    params_ds.save(ds_path)
    Paths.datasets_dict[did] = ds_path
    Paths.save_datasets_dict(Paths.datasets_dict)


