#import PythonModel3dTracker.PyMBVAll as mbv
import json
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as DSI
import PythonModel3dTracker.PythonModelTracker.Landmarks.OpenPoseGrabber as OPG
import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.Paths as Paths

datasets = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04',
            'mhad_s04_a04', 'mhad_s05_a04', 'mhad_s06_a04',
            'mhad_s07_a04', 'mhad_s08_a04', 'mhad_s09_a01',
            'mhad_s09_a02', 'mhad_s09_a03', 'mhad_s09_a04',
            'mhad_s09_a05', 'mhad_s09_a06', 'mhad_s09_a07',
            'mhad_s09_a08', 'mhad_s09_a09', 'mhad_s09_a10',
            'mhad_s09_a11', 'mhad_s10_a04', 'mhad_s11_a04',
            'mhad_s12_a04'
]

op = OPG.OpenPoseGrabber(model_op_path=Paths.models_openpose)
filename_template = Paths.datasets + "human_tracking/mhad/landmark_detections/json_openpose/{}.json"


for dataset in datasets:
    params_ds = DSI.DatasetInfo()
    params_ds.generate(dataset)
    grabber = AutoGrabber.create_di(params_ds)

    ds_data = {
        "frame_idx":[],
        "point_names":[],
        "keypoints3d":[],
        "keypoints2d": [],
        "calibs": [],
        "source": []

    }

    print params_ds.did, 'frame:',
    for f in range(params_ds.limits[0], params_ds.limits[1]+1):
        if f % 50 == 0: print f,
        images, clb = grabber.grab()
        point_names_cur, keypoints3d_cur, keypoints2d_cur, calibs_cur, source_cur = op.acquire(images, clb)
        ds_data['frame_idx'].append(f)
        ds_data['point_names'].append(point_names_cur)
        ds_data['keypoints3d'].append([k3d.__pythonize__() for k3d in keypoints3d_cur])
        ds_data['keypoints2d'].append([k2d.__pythonize__() for k2d in keypoints2d_cur])
        ds_data['calibs'].append(calibs_cur.__pythonize__())
        ds_data['source'].append(source_cur)

    filename = filename_template.format(params_ds.did)
    json_target = open(filename, 'w')
    json.dump(ds_data, json_target, indent=2)
    print('Saving results to: <{}>'.format(filename))




