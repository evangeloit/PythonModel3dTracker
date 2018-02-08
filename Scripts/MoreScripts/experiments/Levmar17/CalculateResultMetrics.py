import json
import os

import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ResultMetricsCalculator as RMC


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/")
output_json = os.path.join(input_dir, 'results_metrics.json')

if __name__ == "__main__":
    results_metrics = RMC.CalculateMetricsDir(input_dir=input_dir)
    with open(output_json, 'w') as fp:
        json.dump(results_metrics, fp)


