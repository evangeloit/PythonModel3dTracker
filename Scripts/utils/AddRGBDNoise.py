import numpy as np
import subprocess

dns_values = np.array([0])
lnr_values = np.array([0.1, 0.2, 0.4])
sta_values = np.arange(0, 651, 50)
fps=30

print lnr_values
print dns_values
print sta_values

for s in sta_values:
    for l in lnr_values:
        full_command = './release/HandTrackerPF/DatasetSerializer -i ds/gtcircsta%03d_fps%d.txt -o ds/gtcircsta%03d_fps%d_lnr%.1f_dns%d.sa -m gt -v 0 --lnr %.1f' % (s,fps,s,fps,l,dns_values[0],l)
        print full_command
        subprocess.call(full_command, shell=True)