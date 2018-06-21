import cv2

#import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.Landmarks.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.Paths as Paths
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru


datasets = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04',
            'mhad_s04_a04', 'mhad_s05_a04', 'mhad_s06_a04',
            'mhad_s07_a04', 'mhad_s08_a04', 'mhad_s09_a01',
            'mhad_s09_a02', 'mhad_s09_a03', 'mhad_s09_a04',
            'mhad_s09_a05', 'mhad_s09_a06', 'mhad_s09_a07',
            'mhad_s09_a08', 'mhad_s09_a09', 'mhad_s09_a10',
            'mhad_s09_a11', 'mhad_s10_a04', 'mhad_s11_a04',
            'mhad_s12_a04'
]
# Unit test.
if __name__ == '__main__':
    # dataset = 'mhad_s03_a04'
    # params_ds = dsi.DatasetInfo()
    # params_ds.generate(dataset)
    # grabber = AutoGrabber.create_di(params_ds)
    # lg = LG.LandmarksGrabber(params_ds=params_ds, params_ds_label='json_openpose')
    #
    # for f in range(10):
    #     images, calibs = grabber.grab()
    #     point_names, keypoints, keypoints2d, clb, _ = lg.acquire(images)
    #     print point_names, keypoints2d[0][0], keypoints[0][0]
    #     viz = images[1]
    #     viz = ru.disp_points(keypoints2d[0], viz)
    #
    #     cv2.imshow("viz",viz)
    #     cv2.waitKey(0)

    for dataset in datasets:
        params_ds = dsi.DatasetInfo()
        params_ds.generate(dataset)
        lg = LG.LandmarksGrabber(params_ds=params_ds, params_ds_label='json_openpose')
        print params_ds.did,
        for f in range(params_ds.limits[0], params_ds.limits[1]):

            lg.seek(f)
            point_names, keypoints, keypoints2d, clb, _ = lg.acquire()
            if f == params_ds.limits[0]:
                init_len = len(point_names)
                print f, init_len,
            if len(point_names) != init_len:
                print f, len(point_names),
        print '.'



