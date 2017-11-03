
import numpy as np
from scipy import signal

# <codecell>

input_res_path = 'rs/dev_01_dev.txt'
output_res_path = 'rs/dev_01_dev_smooth.txt'
res_in = np.loadtxt(input_res_path, dtype=float, comments='#', delimiter=',') 
res_in_first = np.copy(res_in[0])

# <codecell>

trans_in = np.transpose(res_in)
print len(trans_in)

# <codecell>

w = signal.gaussian(7,std=1)
w = w / w.sum()
print w

# <codecell>

trans_out = trans_in
for dim in range(len(trans_in)):
    trans_out[dim] = signal.convolve(trans_in[dim],w,mode='same')

# <codecell>

res_out = np.transpose(trans_out)
res_out[0] = res_in_first
print res_out[0]

# <codecell>

np.savetxt(output_res_path,res_out,delimiter=',')

# <codecell>


