import PythonModel3dTracker.PyMBVALL as mbv
import BlenderMBV.BlenderMBVLib.BlenderMBVConversions as bmc
import PythonModel3dTracker.PythonModelTracker.ModelTrackingGui as GUI
import PythonModel3dTracker.Paths as Paths
### NOT WORKING !!!
model3d = mbv.PF.Model3dMeta.create(Paths.model3d_dict["mh_body_male_custom"]["path"])
bm3d = bmc.BlenderModel3dMeta(model3d,model3d.default_state)


mbv_camera = mbv.Core.CameraMeta()

frame_data = bmc.getFrameDataMBV(model3dmeta=model3d,state=model3d.default_state, frames=[0,10,100])


gui = GUI.ModelTrackingGuiZeromq()

while continue_loop:
    gui_command = gui.recv_command()
    if gui_command.name == "quit":
        continue_loop = False

    if gui_command.name == "state":
        if visualize_params['client'] == 'blender':
            state_gui = gui.recv_state(model3d, state)
            if state_gui is not None:
                state = mbv.Core.DoubleVector(state_gui)
                pf.state = state

    if gui_command.name == "init":
        if visualize_params['client'] == 'blender':
            gui.send_init(blconv.getFrameDataMBV(model3dmeta=model3d, state=state,
                                                 frames=[params_ds.limits[0], f, params_ds.limits[1]],
                                                 scale=0.001))
        else:
            gui_command.name = "frame"

    if gui_command.name == "frame":
        f_gui = gui.recv_frame()
        frame_data = bmc.getFrameDataMBV(model3dmeta=model3d, state=state,
                                                        frames=[params_ds.limits[0], f, params_ds.limits[1])
                                                        )

            gui.send_frame(frame_data)

    if f > params_ds.limits[1]: continue_loop = False

    results.add(f, model3d.model_name, state)

# if res_filename is not None:
#    results.save(res_filename)
return results
# mbv.Core.CachedAllocatorStorage.clear()

# for nn,n in frame_data.blender_model3dmeta.dims.items():
#     for t,v in n.items():
#         print(nn, t,v)
