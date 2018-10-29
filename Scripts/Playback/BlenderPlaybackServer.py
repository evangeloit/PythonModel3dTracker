import os

from PythonModel3dTracker.PythonModelTracker.PlaybackHelper import PlaybackHelper
import PythonModel3dTracker.Paths as Paths

dtname = ['mhad_s06_a01', 'mhad_s06_a02', 'mhad_s06_a03','mhad_s06_a04'\
            , 'mhad_s06_a05', 'mhad_s06_a06','mhad_s06_a07','mhad_s06_a08','mhad_s06_a09', 'mhad_s06_a10', 'mhad_s06_a11']
dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/'

# Input
for name in range(0,len(dtname)):
    wait_time = 1
    dataset = dtpath+dtname[name]+'.json'
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
