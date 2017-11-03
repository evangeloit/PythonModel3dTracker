import os.path

import PythonModelTracker.ModelTrackingResults as mtr
from PythonModelTracker.ResultLandmarksGenerator import GenerateLandmarks

os.chdir(os.environ['bmbv']+'/Scripts/')


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/{0}{1}")
res = ['mhad_s02_a04_mh_body_male_custom_p1_lp1_ransac[0.0, 0.0]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.0, 0.0]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.0, 0.1]',
       'mhad_s02_a04_mh_body_male_custom_p10_lp10_ransac[0.1, 0.2]',
       'mhad_s02_a04_mh_body_male_custom_p20_lp20_ransac[0.1, 0.2]',
       'mhad_s02_a04_mh_body_male_custom_p20_lp20_ransac[0.15, 0.3]',
       'mhad_s02_a04_mh_body_male_custom_p256_lp20_ransac[0.1, 0.2]']

wait_time = 1
for f in res:
    results_in = input_dir.format(f,'.json')
    assert os.path.isfile(results_in)

    results_out = input_dir.format(f,'_out.json')
    results = mtr.ModelTrackingResults()
    results.load(results_in)
    results_ldm = GenerateLandmarks(results)
    results_ldm.save(results_out)





