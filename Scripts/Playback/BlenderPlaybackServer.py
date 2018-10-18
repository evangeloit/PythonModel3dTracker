import os

from PythonModel3dTracker.PythonModelTracker.PlaybackHelper import PlaybackHelper
import PythonModel3dTracker.Paths as Paths

# Input
wait_time = 1
dataset = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/mhad_s12_a04.json'
model_name = 'mh_body_male_customquat'
res = ''
results_txt = None #os.path.join(Paths.results, "Human_tracking/Levmar/mhad_quats/{}.json".format(res))
visualize = {'enable':True,
             'client': 'opencv',
             'labels':True, 'depth':True, 'rgb':True, 'wait_time':1}
assert visualize['client'] in ['opencv','blender']
sel_landmarks = None # "gt" #"gt"   #see dataset json for available landmarks.

# Output options
results_txt_out = None #os.path.join(Paths.datasets, "object_tracking/co4robots/{}_blender.json".format(res))
output_video = None #os.path.join(Paths.datasets,"human_tracking/{}.avi".format(res))
output_frames = None #os.path.join(Paths.results,"Human_tracking/Levmar/mhad_quats/frames/{0}/{0}_{1}.png".format(res,"{:05d}"))

ph = PlaybackHelper(output_video,output_frames)
if results_txt is not None:
    ph.set_results(results_txt,sel_landmarks)
else:
    if model_name is not None: ph.set_model(model_name)
    ph.set_dataset(dataset,sel_landmarks)
ph.playback_loop(visualize)

if results_txt_out is not None:
    ph.results.save(results_txt_out)
