from plyfile import PlyData, PlyElement
import numpy as np

vertex = np.array([(0, 0, 0),
                   (0, 1, 1),
                   (1, 0, 1),
                   (1, 1, 0)],
                   dtype=[('x', 'f4'),
                          ('y', 'f4'),
                          ('z', 'f4')])

el = PlyElement.describe(vertex, 'vertex')
PlyData([el]).write('/home/mad/Development/Projects/htrgbd_scripts/test.ply')