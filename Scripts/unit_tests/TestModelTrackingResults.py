import PythonModelTracker.ModelTrackingResults as mtr


results_txt = "/home/mad/Development/Results/Human_tracking/Levmar/mhad_s02_a04_mh_body_male_custom_p1_lp1_ransac[0.0, 0.0].json"

results = mtr.ModelTrackingResults()
results.load(results_txt)

print 'Frame range:',results.get_limits()
print 'Models: ',results.models
for m in results.models:
    states = results.get_model_states(m)
    landmark_names, landmarks = results.get_model_landmarks(m)
    print landmark_names
    for f in range(10):
        if f in states: print f, states[f]
        if f in landmarks: print f, landmarks[f]
