import os

import cv2

os.chdir(os.environ['bmbv']+"/Scripts/")
accepted_keys = ['q', 'p', 'n']

limits = [2, 200]
res = ['openpose',
       'mhad_s02_a04_mh_body_male_custom_p1_lp1_ransac[0.0, 0.0]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.0, 0.0]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.0, 0.1]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.1, 0.2]',
       'mhad_s02_a04_mh_body_male_custom_p20_lp20_ransac[0.1, 0.2]',
       'mhad_s02_a04_mh_body_male_custom_p20_lp20_ransac[0.15, 0.3]',
       'mhad_s02_a04_mh_body_male_custom_p256_lp20_ransac[0.1, 0.2]']
res_folders = [os.path.join(Paths.results, "Human_tracking/Levmar/{0}/{1}.png".format(r, "{:05d}")) for r in res]


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

