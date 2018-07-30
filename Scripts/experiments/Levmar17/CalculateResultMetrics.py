import json
import os

import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ResultMetricsCalculator as RMC


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/mhad_quats/v6")
output_json = os.path.join(Paths.results, "Human_tracking/Levmar/mhad_quats/v6_metrics_new.json")

if __name__ == "__main__":
    results_metrics = RMC.CalculateMetricsDir(input_dir=input_dir)
    with open(output_json, 'w') as fp:
        json.dump(results_metrics, fp)


