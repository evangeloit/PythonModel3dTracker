import os

from PythonModel3dTracker.PythonModelTracker.PlaybackHelper import PlaybackHelper
import PythonModel3dTracker.Paths as Paths

# Input
wait_time = 1
dataset = 'mhad_s01_a04'
model_name = 'mh_body_male_custom_vector'
res = 'boxtalosreem_01_gt'
results_txt = None #os.path.join(Paths.datasets, "object_tracking/co4robots/{}.json".format(res))
visualize = {'enable':True,
             'client': 'blender',
             'labels':True, 'depth':True, 'rgb':True, 'wait_time':0}
assert visualize['client'] in ['opencv','blender']
sel_landmarks = None # "gt" #"gt"   #see dataset json for available landmarks.

# Output options
results_txt_out = None #os.path.join(Paths.datasets, "object_tracking/co4robots/{}_blender.json".format(res))
output_video = None #os.path.join(paths.datasets,"object_tracking/co4robots/{}.avi".format(res))
output_frames = None #pos.path.join(Paths.datasets,"human_tracking/co4robots/ms1_gestures/frames/s4_{1}.png".format(res,"{:05d}"))

ph = PlaybackHelper(output_video,output_frames)
if results_txt is not None:
    ph.set_results(results_txt,sel_landmarks)
else:
    if model_name is not None: ph.set_model(model_name)
    ph.set_dataset(dataset,sel_landmarks)
ph.playback_loop(visualize)

if results_txt_out is not None:
    ph.results.save(results_txt_out)
