import PyMBVCore as core
import numpy as np
from scipy import signal
import csv
import re


def load_states(results_txt, n_dims):
    states = []
    with open(results_txt, 'r') as csvfile:
        results_reader = csv.reader(csvfile, delimiter=',')
        for f,row in enumerate(results_reader):
            cur_state_list = [float(re.sub('[\[\]]','',i)) for i in row]
            assert len(cur_state_list) == n_dims
            states.append(core.DoubleVector(cur_state_list))
            #print f,':', ','.join(row)
    return states

def smooth_states(states,window_size=7,std=1):
    n_frames = len(states)
    n_dims = len(states[0])
    states_array = np.zeros((n_frames,n_dims))
    for s,state in enumerate(states):
        for d in range(n_dims):
            states_array[s, d] = state[d]
    states_array = np.transpose(states_array)

    w = signal.gaussian(window_size, std)
    w = w / w.sum()

    for dim in range(n_dims):
        states_array[dim] = signal.convolve(states_array[dim], w, mode='same')

    for s,state in enumerate(states):
        for d in range(n_dims):
            state[d] = states_array[d, s]

    return states