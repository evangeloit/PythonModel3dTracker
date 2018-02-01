import numpy as np
import cv2
import copy

import PythonModel3dTracker.PyMBVAll as mbv
import BlenderMBV.BlenderMBVLib.AngleTransformations as at




def ExtractPosition(state):
    return state[0:3]


def ExtractPositionMBV(state):
    return mbv.Core.Vector3(state[0:3])


def SetPosition(state, position):
    state[0] = np.float64(position[0])
    state[1] = np.float64(position[1])
    state[2] = np.float64(position[2])
    return state


def SetPositionRotationRt(state, Rt):
    _, _, angles, tr, _ = at.decompose_matrix(Rt)
    quat = at.quaternion_from_euler(angles[0], angles[1], angles[2])
    state[0] = tr[0]
    state[1] = tr[1]
    state[2] = tr[2]
    state[3] = quat[1]
    state[4] = quat[2]
    state[5] = quat[3]
    state[6] = quat[0]
    # print 'State:', state
    return state


def SetPositionRotationRvecT(default_state, rtvec):
    state = copy.deepcopy(default_state)
    R, _ = cv2.Rodrigues(np.array([rtvec[0], rtvec[1], rtvec[2]]))

    quat = at.quaternion_from_matrix(R)
    #print 'Rmat:\n', R
    #print 't:', rtvec[3], rtvec[4], rtvec[5]
    state[0] = rtvec[3]
    state[1] = rtvec[4]
    state[2] = rtvec[5]
    state[3] = quat[1]
    state[4] = quat[2]
    state[5] = quat[3]
    state[6] = quat[0]
    # print 'State:', state

    return state


def ExtractQuaternion(state):
    # Output Quat: [w, x, y, z]
    return [state[6], state[3], state[4], state[5]]


def ExtractQuaternionMBV(state):
    return mbv.Core.Quaternion(x=state[3], y=state[4], z=state[5], w=state[6])


def SetQuaternion(state, quat):
    # Input Quat: [w, x, y, z]
    state[3] = np.float64(quat[1])
    state[4] = np.float64(quat[2])
    state[5] = np.float64(quat[3])
    state[6] = np.float64(quat[0])


def ExtractPoseMatrix(state):
    position = ExtractPosition(state)
    quaternion = ExtractQuaternion(state)
    pose = at.compose_matrix(translate=position, angles=at.euler_from_quaternion(quaternion))
    return pose


def ExtractPoseMatrixMBV(state):
    pose = ExtractPoseMatrix(state)
    pose_mbv = mbv.Core.Matrix4x4(pose.astype(np.float32).T)
    return pose_mbv


def TransformPose(state, tf_matrix):
    pose = ExtractPoseMatrixMBV(state)
    #print 'state:\n', state
    #print 'tf_matrix:\n', tf_matrix
    #print 'pose:\n', pose
    transformed_pose = tf_matrix * pose
    transformed_pose_np = transformed_pose.data.T
    transformed_quat = at.quaternion_from_matrix(transformed_pose_np)
    transformed_position = at.translation_from_matrix(transformed_pose_np)

    transformed_state = copy.deepcopy(state)
    SetPosition(transformed_state, transformed_position)
    SetQuaternion(transformed_state, transformed_quat)
    return transformed_state
