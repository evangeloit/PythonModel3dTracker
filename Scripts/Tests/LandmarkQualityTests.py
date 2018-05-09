import cv2
import os
import numpy as np

import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.OpenPoseGrabber as opg
import PythonModel3dTracker.PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.DepthMapUtils as DMU
import PythonModel3dTracker.PythonModelTracker.Model3dUtils as M3DU
import PythonModel3dTracker.PythonModelTracker.GeomUtils as GU
import PythonModel3dTracker.Paths as Paths
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru





# Unit test.
if __name__ == '__main__':
    interpolate_bones =  [
        "R.UArm",
        "R.LArm",
        "R.ULeg",
        "R.LLeg",
        "L.UArm",
        "L.LArm",
        "L.ULeg",
        "L.LLeg"
    ]
    model_name = "mh_body_male_custom"
    model3d_xml = os.path.join(Paths.models, Paths.model3d_dict[model_name]['path'])

    model3d = mbv.PF.Model3dMeta.create(str(model3d_xml))
    model3d.parts.genBonesMap()
    dataset = 'mhad_s01_a04'
    params_ds = dsi.DatasetInfo()
    params_ds.generate(dataset)
    grabber = AutoGrabber.create_di(params_ds)
    op = opg.OpenPoseGrabber(model_op_path=Paths.models_openpose)

    for f in range(10):
        grabber.seek(5*f)
        images, calibs = grabber.grab()
        point_names, keypoints3d, keypoints2d, clb, ldm_source = op.acquire(images, calibs)
        dpt = images[0]
        rgb = images[1]

        #kp_depths = DMU.GetMedianDepths(keypoints2d[0], dpt, 4)
        point_names_, keypoints2d_ =\
            M3DU.GetInterpKeypointsModelSets(landmark_source=ldm_source,
                                             model3d=model3d,
                                             point_names=point_names,
                                             keypoints2d=keypoints2d[0],
                                             interpolate_set=interpolate_bones)
        keypoints3d_ = DMU.UnprojectPointSets(keypoints2d_, calibs[0], dpt)

        _, keypoints2d_all = \
            M3DU.GetInterpKeypointsModel(landmark_source=ldm_source,
                                         model3d=model3d,
                                         point_names=point_names,
                                         keypoints2d=keypoints2d[0],
                                         interpolate_set=interpolate_bones)



        for n, p3d, p2d in zip(point_names_, keypoints3d_, keypoints2d_):
            if len(p3d)>2:
                line_dist = GU.NormalizedLineDist(p3d[0], p3d[-1],p3d[2:-1],50)
            else:
                line_dist = 0



            print n[0], '\t', line_dist, p2d


        viz = ru.disp_points(keypoints2d_all, rgb)

        cv2.imshow("viz",viz)
        cv2.waitKey(0)