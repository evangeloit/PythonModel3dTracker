
# Dictionary of landmark correspondences between: (landmard_detection_source, skinned_model).
primitives_dict = {
    ("damien", "human_ext"):
    {"head":"head_sphere_0","neck":"neck_sphere_0","bodyCenter":"body_sphere_1",
     "hip":"body_sphere_2","leftShoulder":"arm_left_sphere_0","rightShoulder":"arm_right_sphere_0",
     "leftElbow":"arm_left_sphere_1","rightElbow":"arm_right_sphere_1","leftWrist":"arm_left_sphere_2",
     "rightWrist":"arm_right_sphere_2","leftLegRoot":"leg_left_sphere_0",
     "rightLegRoot":"leg_right_sphere_0","leftKnee":"leg_left_sphere_1","rightKnee":"leg_right_sphere_1",
     "leftAnkle":"leg_left_sphere_2","rightAnkle":"leg_right_sphere_2"},
    ("damien", "mh_body_male"):
    {"head":"head","neck":"neck","bodyCenter":"spine-1",
     "hip":"hips","leftShoulder":"deltoid.L","rightShoulder":"deltoid.R",
     "leftElbow":"forearm.L","rightElbow":"forearm.R","leftWrist":"hand.L",
     "rightWrist":"hand.R","leftLegRoot":"thigh.L",
     "rightLegRoot":"thigh.R","leftKnee":"shin.L","rightKnee":"shin.R",
     "leftAnkle":"foot.L","rightAnkle":"foot.R"},
    ("roditak", "hand_skinned_rds"):
    {'f_pinky.03.R':'f_pinky.03.R',
     'f_middle.03.R':'f_middle.03.R',
     'f_ring.03.R':'f_ring.03.R',
     'thumb.03.R':'thumb.03.R',
     'f_index.03.R':'f_index.03.R'},
    ("COCO", "mh_body_male_custom"):
    {'L.LLeg': 'L.LLeg',
     'L.ULeg': 'L.ULeg',
     'R.LLeg': 'R.LLeg',
     'R.Foot': 'R.Foot',
     'R.LArm': 'R.LArm',
     'R.eye': 'R.eye',
     'L.LArm': 'L.LArm',
     'L.Wrist': 'L.Wrist',
     'R.ULeg': 'R.ULeg',
     'R.ear': 'R.ear',
     'L.ear': 'L.ear',
     'L.eye': 'L.eye',
     'Nose': 'Nose',
     'L.UArm': 'L.UArm',
     'neck': 'neck',
     'R.Wrist': 'R.Wrist',
     'R.UArm': 'R.UArm',
     'L.Foot': 'L.Foot'},
    ("bvh", "mh_body_male_custom"):
        {'LeftUpLeg': 'L.ULeg', 'LeftLeg': 'L.LLeg', 'LeftFoot': 'L.Foot',
         'RightUpLeg': 'R.ULeg', 'RightLeg': 'R.LLeg', 'RightFoot': 'R.Foot',
         'LeftArm': 'L.UArm', 'LeftForeArm': 'L.LArm', 'LeftHand': 'L.Wrist',
         'RightArm': 'R.UArm', 'RightForeArm': 'R.LArm', 'RightHand': 'R.Wrist',
         'Neck': 'neck.001', 'spine': 'root'
        },
    ("csv", "mh_body_male_custom"):
    {"head":"neck.001","neck":"neck","bodyCenter":"root",
     "leftShoulder":"L.UArm", "leftElbow":"L.LArm", "leftWrist":"L.Wrist",
     "rightShoulder":"R.UArm", "rightElbow":"R.LArm", "rightWrist":"R.Wrist",
     "leftLegRoot":"L.ULeg", "leftKnee":"L.LLeg","leftAnkle":"L.Foot",
     "rightLegRoot":"R.ULeg","rightKnee":"R.LLeg","rightAnkle":"R.Foot"
     },


    # {'LeftLeg': 'L.LLeg', 'LeftUpLeg': 'L.ULeg', 'RightLeg': 'R.LLeg', 'RightFoot': 'R.Foot', 'RightForeArm': 'R.LArm',
    #  'LeftShoulder': 'L.shoulder', 'LeftForeArm': 'L.LArm',
    #  'RightShoulder': 'R.shoulder', 'LeftHand': 'L.Wrist', 'RightUpLeg': 'R.ULeg',
    #   'Head': 'neck.001', 'LeftArm': 'L.UArm', 'Neck': 'neck.001', 'spine': 'root',
    #  'RightHand': 'R.Wrist', 'RightArm': 'R.UArm', 'LeftFoot': 'L.Foot'}

    # 'Hips', 'spine', 'spine1', 'spine2',
    # 'Neck', 'Head', 'Site', 'RightShoulder',
    # 'RightArm', 'RightArmRoll', 'RightForeArm',
    # 'RightForeArmRoll', 'RightHand', 'Site', 'LeftShoulder',
    # 'LeftArm', 'LeftArmRoll', 'LeftForeArm', 'LeftForeArmRoll',
    # 'LeftHand', 'Site', 'RightUpLeg', 'RightUpLegRoll', 'RightLeg',
    # 'RightLegRoll', 'RightFoot', 'RightToeBase', 'Site', 'LeftUpLeg',
    # 'LeftUpLegRoll', 'LeftLeg', 'LeftLegRoll', 'LeftFoot', 'LeftToeBase', 'Site'

# {'LeftLeg': 'L.LLeg', 'LeftUpLeg': 'L.ULeg', 'RightLeg': 'R.LLeg', 'RightFoot': 'R.Foot', 'RightForeArm': 'R.LArm',
#      'Head': 'R.eye', 'LeftShoulder': 'L.shoulder', 'spine2': 'R.torso', 'LeftForeArm': 'L.LArm',
#      'RightShoulder': 'R.shoulder', 'LeftHand': 'L.Wrist', 'RightUpLeg': 'R.ULeg', 'Head': 'R.ear', 'Head': 'L.ear',
#      'Head': 'L.eye', 'Head': 'Nose', 'LeftArm': 'L.UArm', 'Neck': 'neck', 'Neck': 'neck.001', 'spine': 'root',
#      'RightHand': 'R.Wrist', 'spine2': 'L.torso', 'RightArm': 'R.UArm', 'LeftFoot': 'L.Foot'}
}
primitives_dict[("damien", "human_ext_collisions")] = primitives_dict[("damien", "human_ext")]
primitives_dict[("damien", "mh_body_male_meta")] = primitives_dict[("damien", "mh_body_male")]
primitives_dict[("damien", "mh_body_male_meta_grpscl")] = primitives_dict[("damien", "mh_body_male")]
primitives_dict[("COCO", "mh_body_male_customquat")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_vector")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_meta")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_0850")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_0900")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_0950")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_1050")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_1100")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_1150")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("COCO", "mh_body_male_custom_meta_glbscl")] = primitives_dict[("COCO", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_meta")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_0850")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_0900")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_0950")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_1050")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_1100")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_1150")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_customquat")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("bvh", "mh_body_male_custom_vector")] = primitives_dict[("bvh", "mh_body_male_custom")]
primitives_dict[("csv", "mh_body_male_custom_0950")] = primitives_dict[("csv", "mh_body_male_custom")]
primitives_dict[("csv", "mh_body_male_customquat")] = primitives_dict[("csv", "mh_body_male_custom")]
primitives_dict[("csv", "mh_body_male_custom_vector")] = primitives_dict[("csv", "mh_body_male_custom")]

# AMMAR Synthetic MHAD Correspondences.
# landmark_names = [
#  "L.UArm", "L.LArm", "L.Wrist",
#  "R.UArm", "R.LArm", "R.Wrist",
#  "L.ULeg", "L.LLeg", "L.Foot",
#  "R.ULeg", "R.LLeg", "R.Foot",
#  "neck.001", "neck", "root"
# ]
#
# landmark_positions = [ [0,0,0], [0,0,0], [0,0,0],
#                        [0,0,0], [0,0,0], [0,0,0],
#                        [0,150,0], [0,0,0], [0,0,0],
#                        [0,150,0], [0,0,0], [0,0,0],
#                        [0,120,0], [0,0,0], [0,0,0]]


# Model positions (local) of landmarks for each (landmard_detection_source, skinned_model) pair.
model_landmark_positions = {
    ("csv", "mh_body_male_custom"):
        {"rightLegRoot": [0, 350, 0], "rightWrist": [0, 70, 0],
         "head": [0, -20, 0], "bodyCenter": [0, 0, 0],
         "rightKnee": [0, 350, 0], "leftShoulder": [0, 200, 0],
         "leftKnee": [0, 350, 0], "rightShoulder": [0, 200, 0],
         "leftLegRoot": [0, 350, 0], "leftWrist": [0, 70, 0]},
    ("bvh", "mh_body_male_custom"):
        {'LeftUpLeg': [0, 0, 0], 'LeftLeg': [0, 0, 0], 'LeftFoot': [0, 0, 0],
         'RightUpLeg':[0, 0, 0], 'RightLeg': [0, 0, 0], 'RightFoot': [0, 0, 0],
         'LeftArm': [0,0,0], 'LeftForeArm': [0, 0, 0], 'LeftHand': [0,0,0],
         'RightArm': [0,0,0], 'RightForeArm': [0, 0, 0], 'RightHand': [0,0,0],
         'Neck': [0,-20,0], 'spine': [0,0,0]
        }
}
model_landmark_positions[("bvh", "mh_body_male_custom_vector")] = model_landmark_positions[("bvh", "mh_body_male_custom")]
model_landmark_positions[("bvh", "mh_body_male_customquat")] = model_landmark_positions[("bvh", "mh_body_male_custom")]


# Maps observation landmarks to model partitions
# for each (landmard_detection_source, skinned_model) pair.
model_landmark_partitions = {
    ("bvh", "mh_body_male_custom"):
        {'LeftUpLeg': "global_pos", 'LeftLeg': "l_leg", 'LeftFoot': "l_leg",
         'RightUpLeg':"global_pos", 'RightLeg': "r_leg", 'RightFoot': "r_leg",
         'LeftArm': "global_pos", 'LeftForeArm': "l_arm", 'LeftHand': "l_arm",
         'RightArm': "global_pos", 'RightForeArm': "r_arm", 'RightHand': "r_arm",
         'Neck': "head", 'spine': "global_pos"
        }
}
model_landmark_partitions[("bvh", "mh_body_male_custom_vector")] = model_landmark_partitions[("bvh", "mh_body_male_custom")]
model_landmark_partitions[("bvh", "mh_body_male_customquat")] = model_landmark_partitions[("bvh", "mh_body_male_custom")]
