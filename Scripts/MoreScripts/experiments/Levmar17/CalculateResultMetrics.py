import numpy as np
import os

import PyModel3dTracker as mt

from PythonModelTracker.ModelTrackingResults import ModelTrackingResults

input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/{0}{1}")
res = ['mhad_s02_a04_mh_body_male_custom_p1_lp1_ransac[0.0, 0.0]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.0, 0.0]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.0, 0.1]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.1, 0.2]',
       'mhad_s02_a04_mh_body_male_custom_p20_lp20_ransac[0.1, 0.2]',
       'mhad_s02_a04_mh_body_male_custom_p20_lp20_ransac[0.15, 0.3]',
       'mhad_s02_a04_mh_body_male_custom_p256_lp20_ransac[0.1, 0.2]']


# Load landmarks
res_filename1 = input_dir.format(res[0],'.json')
res1 = ModelTrackingResults()
res1.load(res_filename1)

res_filename2 = input_dir.format(res[1],'.json')
res2 = ModelTrackingResults()
res2.load(res_filename2)

# Create LandmarkDistObjective
metric_calculator = mt.LandmarksDistObjective()

model_name = res1.models[0]
lnames, landmarks1 = res1.get_model_landmarks(model_name)
lnames, landmarks2 = res2.get_model_landmarks(model_name)
assert len(landmarks1) == len(landmarks2)

for f1,f2 in zip(landmarks1, landmarks2):
    assert f1 == f2
    lnp1 = np.array(landmarks1[f1])
    lnp2 = np.array(landmarks2[f2])
    dists = np.linalg.norm(lnp1-lnp2,axis=1)
    avg_dist = np.average(dists)
    print f1, avg_dist
