import numpy as np
import itertools as it
import sys
import csv

import PFFileIO as pfio
import AngleTransformations as at
from numpy.f2py.auxfuncs import isarray

synthetic_dataset_filename='../media/pf/dataset/test_dataset.txt'
p=pfio.load_vectors('../media/pf/pf_config.xml','main')
quaternions_filename = '../media/pf/dataset/quaternions_256.csv'
n_rotations=50  #number of quaternions to load. -1 to load all from 'quaternions_filename'.
n_rsamples=10
var_fraction = 1
quat_var_fraction = 0.1

def generate_grid_samples(n_steps,p):
    values_step=(p.highBounds-p.lowBounds)/(n_steps-1)
    print 'Values Step: \n',values_step
    param_values = list()
    for idx, n in enumerate(values_step):
        param_values.append(list())
        #Use low-high bounds for all other variables.
        if ((idx<=2)|(idx>=7)):
            if p.sel_dims[idx] == 1:
                for v in range(n_steps):
                    param_values[idx].append(p.lowBounds[idx]+v*values_step[idx])
            else:
                param_values[idx].append(p.init_state[idx])
            
    print param_values
    
    prd =it.product(param_values[0],param_values[1],param_values[2],
                    param_values[3],param_values[4],param_values[5],
                    param_values[6],param_values[7],param_values[8],
                    param_values[9],param_values[10],param_values[11],
                    param_values[12],param_values[13],param_values[14],
                    param_values[15],param_values[16],param_values[17],
                    param_values[18],param_values[19],param_values[20],
                    param_values[21],param_values[22],param_values[23],
                    param_values[24],param_values[25],param_values[26])
    return prd
    
    

def load_quaternions(input_file,n_rotations):
    """Loads up to 'n_rotations' quaternions from a csv file (w,x,y,z)
    quat_prd: the returned quaternions as a list of string lists.
    """
    f = open(input_file, 'r')
    reader = csv.reader(f, delimiter=',')
    quat_prd = list()
    for idx, row in enumerate(reader):
        if (n_rotations>0):
            if (idx > n_rotations):
                break
        #first_elem=row.pop(0)
        #row.append(first_elem)
        quat_prd.append(row)      
    
    return quat_prd
    
    

def discretize_rotations(n_steps):
    """Rotation Discretization in to n_steps using Euler angles. ()
    TODO:Fix the problem with duplicate rotations.
    (The current version of the function is not working properly.
    """
    #use euler for quaternions
    euler_step = 1.57
    euler_values = list()
    for idx in range(3):
        euler_values.append(list())
        for v in range(n_steps):
            euler_values[idx].append(v*euler_step)
    
    euler_values[1].pop()
    euler_values[1].pop()
    euler_values[1].append(3*1.57)
    
    print 'euler_values:', euler_values
    euler_prd = it.product(euler_values[0],euler_values[1],euler_values[2])    
    quat_prd = list()
    for i in euler_prd:
        if ((i[0]==0) or (i[1]==0) or (i[2]==0)):
            quat = at.quaternion_from_euler(i[0], i[1], i[2], axes='rxyz')
            quat_prd.append(quat)
            print 'euler_prd:',i, '  euler_quat:', quat
    return quat_prd


  
def gen_random_samples_from_rotation(quat,n_samples,var_fraction,quat_var_fraction,p):
    """Generates random samples for a particular 3d rotation defined in quat.
    """
    interval = var_fraction*(p.highBounds - p.lowBounds)
    uniform_noise = np.zeros([n_samples,interval.size])
    for j in range(interval.size):
        uniform_noise[:,j] = np.random.uniform(0, interval[j], n_samples)
     
    random_samples=list()   
    for i in range(n_samples):
        cur_sample = uniform_noise[i,:]+np.array(p.lowBounds)
        cur_sample[0:3] = p.init_state[0:3]
        cur_sample[3:7] = quat+quat_var_fraction*uniform_noise[i,3:7]
        cur_sample = np.minimum(cur_sample,p.highBounds)
        cur_sample = np.maximum(cur_sample,p.lowBounds)
        random_samples.append(cur_sample)
        
    #print 'random_samples:', random_samples
    return random_samples
  


quat_prd = load_quaternions(quaternions_filename,n_rotations)
#quat_prd = discretize_rotations(4)
samples=list()
for r in quat_prd:
    samples.append(gen_random_samples_from_rotation(np.array(r).astype(float),n_rsamples,var_fraction,quat_var_fraction,p))


f = open(synthetic_dataset_filename, 'w')
writer = csv.writer(f)
for r,sample in enumerate(samples):
    for i in sample:
        conc_i=list()
        for j in i:
            if (type(j) is np.ndarray):
                for k in j:
                    conc_i.append(k)
            else:
                conc_i.append(j)
        conc_i.append(r)
        writer.writerow( conc_i )    
f.close()






