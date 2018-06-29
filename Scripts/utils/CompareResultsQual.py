import os
import PythonModel3dTracker.Paths as Paths
import cv2

os.chdir(os.environ['bmbv']+"/Scripts/")
accepted_keys = ['q', 'p', 'n']

limits = [2, 200]
res = ['mhad_s03_a04_mh_body_male_customquat_p1_lp1_ransac[0.0, 0.0]_foFalse_fhFalse',
       'mhad_s03_a04_mh_body_male_customquat_p1_lp1_ransac[0.0, 0.0]_foFalse_fhTrue',
       'mhad_s03_a04_mh_body_male_customquat_p1_lp1_ransac[0.0, 0.0]_foTrue_fhFalse',
       'mhad_s03_a04_mh_body_male_customquat_p1_lp1_ransac[0.0, 0.0]_foTrue_fhTrue',
       ]
res_folders = [os.path.join(Paths.results, "Human_tracking/Levmar/mhad_quats/frames/{0}/{0}_{1}.png".format(r, "{:05d}")) for r in res]


f = limits[0]
while True:
    print 'Frame ', f
    for rn,r in zip(res,res_folders):
        fn = r.format(f)
        if os.path.isfile(fn):
            im = cv2.imread(fn)
            cv2.imshow(rn,im)
        else: print 'Warning: file {} missing'.format(fn)

    key = 'a'
    while not (key in accepted_keys):
        key = chr(cv2.waitKey(0) & 255)

    if key == 'q': break
    if key == 'p': f -= 1
    if key == 'n': f += 1
    if (f > limits[1]) or (f < limits[0]): break

