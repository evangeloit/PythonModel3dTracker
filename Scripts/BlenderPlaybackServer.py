import os

from PythonModel3dTracker.Scripts.PlaybackHelper import PlaybackHelper
import PythonModel3dTracker.Paths as Paths

# Input
wait_time = 1
dataset = None#os.path.join(paths.datasets, "object_tracking/co4robots/{}.oni".format("box_eps_02"))
model_name = "box" #"hand_skinned"#"mh_body_male_meta_glbscl"
res = 'box_eps_01'
results_txt = os.path.join(Paths.datasets, "object_tracking/co4robots/{}_gt.json".format(res))
visualize = {'enable':True,
             'client': 'blender',
             'labels':True, 'depth':True, 'rgb':True, 'wait_time':0}
assert visualize['client'] in ['opencv','blender']
sel_landmarks = None #"gt"   #see dataset json for available landmarks.

# Output options
results_txt_out = results_txt# os.path.join(paths.datasets, "object_tracking/co4robots/{}_new.json".format(res))
output_video = None#os.path.join(paths.datasets,"object_tracking/co4robots/{}.avi".format(res))
output_frames = None #os.path.join(paths.results,"Human_tracking/Levmar/{0}/{1}.png".format(res,"{:05d}"))

ph = PlaybackHelper(output_video,output_frames)
if results_txt is not None:
    ph.set_results(results_txt,sel_landmarks)
else:
    if model_name is not None: ph.set_model(model_name)
    ph.set_dataset(dataset,sel_landmarks)
ph.playback_loop(visualize)

if results_txt_out is not None:
    ph.results.save(results_txt_out)
