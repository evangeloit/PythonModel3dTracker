import pylab as pl
import numpy as np
import csv

#Setting input gt file.
input_fps = 30
input_gt_path ='ds/gt.txt'

output_fps = 5
output_gt_path='ds/gt_%dfps.txt' % output_fps

#Loading input vectors from csv
print 'Loading input vectors from <%s>.' % input_gt_path
gt_in = np.loadtxt(input_gt_path, dtype=float, comments='#', delimiter=' , ')  


n_frames_input = gt_in.shape[0]
n_dims = gt_in.shape[1]
print 'n_frames_input: ', n_frames_input, ' n_dims: ', n_dims

input_step = 1.0 / float(input_fps)
output_step = 1.0 / float(output_fps)
duration_in = 0.000001 + ( float(n_frames_input-1) / float(input_fps))
duration_out = 0.000001 + duration_in - (duration_in % output_step)
n_frames_output = int(duration_out * float(output_fps)) + 1

print 'Input fps: %i. step: %.3f  frames: %i ' % (input_fps, input_step , n_frames_input)
print 'Output fps: %i. step: %.3f  frames: %i ' % (output_fps, output_step , n_frames_output)
print 'Input duration: %.3f Output duration: %.3f ' % (duration_in, duration_out)

in_timestamps = np.arange(0, duration_in, input_step)
out_timestamps = np.arange(0, duration_out, output_step)
#print in_timestamps
#print out_timestamps

gt_out = np.empty([n_frames_output,n_dims],dtype=float)
in_counter = 0
eor_in = 0
in_t1=in_timestamps[in_counter]
in_t2=in_timestamps[in_counter+1]
for idx,out_t in enumerate(out_timestamps): 
    if (eor_in==0):   
        while in_timestamps[in_counter+1]<out_t:
            in_counter = in_counter + 1
            in_t1 = in_timestamps[in_counter]
            in_t2 = in_timestamps[in_counter+1] 
            if in_counter+1>=in_timestamps.size-1:
                eor_in = 1;           
                break
    
    #print 'out_t:%.3f [%.3f,%.3f]' %  (out_t,in_t1,in_t2),
    w1 = abs(1-((out_t-in_t1)/input_step))
    w2 = abs(1-w1)
    #print ' weights:[%.3f,%.3f]' % (w1,w2)
    gt_out[idx] = w1*gt_in[in_counter]+w2*gt_in[in_counter+1]
    #print gt_out[idx]

#Saving output vectors to csv.
print 'Saving %d output vectors to <%s>.' % (n_frames_output,output_gt_path)
np.savetxt(output_gt_path, gt_out, fmt='%.18e', delimiter=' , ')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    