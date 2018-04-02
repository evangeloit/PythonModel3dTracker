import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVParticleFilter as pf
import PyMBVOpenMesh as mbvom
import numpy as np

np.set_printoptions(precision=1)
import cv2
import copy
import matplotlib.pyplot as plt
import RenderingUtils as ru
import AngleTransformations as at

def DecodeToTranslations(decoding):
    hyp = []
    tr = []
    for d in decoding:
        n = len(d.data().matrices)
        cur_tr = np.zeros((3,n))
        cur_hyp = []
        for i, (hid,m) in enumerate(zip(d.data().ids, d.data().matrices)):
            dec_p,dec_s,dec_q = core.DecomposeMatrix(m)
            cur_tr[0,i] = dec_p.x
            cur_tr[1,i] = dec_p.y
            cur_tr[2,i] = dec_p.z
            cur_hyp.append(hid)
        hyp.append(cur_hyp)
        tr.append(cur_tr)
    return hyp,tr




model_xml = "models3d_samples/hand_std/hand_std.xml"
model3d = pf.Model3dMeta.create(model_xml)
state = model3d.default_state

n_bones = 0
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
decoder = dec.GenericDecoder()
print 'Loading model from <', model3d.model_collada, '>'
decoder.loadFromFile(model3d.model_collada,False)
decoder.loadMeshTickets(mmanager)

steps = 1
for i in range(steps):
    # Setting param Vector

    # state[0] += 10
    # state[1] += 10
    state[2] = 700
    rot_q = at.quaternion_from_euler(0,0,3.14)
    state[3] = rot_q[1]
    state[4] = rot_q[2]
    state[5] = rot_q[3]
    state[6] = rot_q[0]
    state[7] += 0.1

    state2 = copy.deepcopy(state)
    state2[2] = 900

    multi_state = core.ParamVectors([state]+[state2])
    decoding = decoder.quickDecode(multi_state)
    hid,translations = DecodeToTranslations(decoding)
    print 'Decoding extractor, h_num: ', pf.DecodingExtractor.getHypothesesNum(decoding)
    #print 'Decoding extractor, matrix(h2,p4): ', pf.DecodingExtractor.getDecodingMatrix(1,decoding,pf.IssueInstanceLabel(2,4)).data
    for d,hmesh,tr in zip(decoding,hid,translations):
        mesh_filename = mmanager.getMeshFilename(d.key())
        print 'Mesh {0} translations:\n'.format(mesh_filename)
        for j,h in enumerate(hmesh):
            print h,tr[:,j].T
