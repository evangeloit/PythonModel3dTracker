import json
import os

import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ResultMetricsCalculator as RMC


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/mhad_quats/v6")
output_json = os.path.join(Paths.results, "Human_tracking/Levmar/mhad_quats/v6_metrics.json")

if __name__ == "__main__":
    if os.path.isfile(output_json):
        with open(output_json, 'r') as fp:
            saved_metrics = json.load(fp)
    else: saved_metrics = None
    results_metrics = RMC.CalculateMetricsDir(input_dir=input_dir, saved_metrics=saved_metrics)
    with open(output_json, 'w') as fp:
        json.dump(results_metrics, fp)


