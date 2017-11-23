### Learn Object Appearance

The following procedure creates a dataset with representative object poses, 
sets the groundtruth manually and learns its appearance by extracting ORB
descriptors and registering them to the 3d object model. The appearance 
can be subsequently used to detect the object in an RGB frame.

1. Run `CaptureDataset` to get a dataset of representative object poses.
   - Set the dataset id: `did`.
   - Set the model name: `model_name`
2. Run `BlenderPlaybackServer` to set the groundtruth of the dataset.
   - Set the dataset id of Step 1.
   - Set the `res` `results_txt` to read the default groundtruth, usually `{did}_gt.json`.
   - Set the `results_txt_out` to write the groundtruth to. Default is `results_txt`.
   - Set `visualize {client:'blender'}`
   - Open blender and run the `mbv_connect` command.
   - For each frame perform the following actions in Blender:
     - Register the object to the point cloud.
     - Run the `mbv_sendstate` command.
   - Eventually close the connection by running: `mbv_quit` 
3. Run `LearnModel3dAppearance` to extract the appearance descriptors and save them.
  - Set the `did` of Step 1.
  - Set the output filename `appearance_filename`.
  - Set the list of dataset frames that will be used: `frames`.