import os

import PyMBVCore as core
import PyMBVDecoding as dec

os.chdir(os.environ['bmbv']+"/Scripts/")


# model_xml = paths.model3d_dict['hand_skinned_meta'][1]
# model3d = pf.Model3dMeta.create(model_xml)
# for d in range(model3d.n_dims):
#     print(d, model3d.dim_names[d],model3d.dim_types[d], model3d.dim_axes[d], \
#         model3d.low_bounds[d], model3d.default_state[d], model3d.high_bounds[d])
#
# #print len(model3d.state_transforms.process(model3d.low_bounds,dec.StateTransformDirection.Forward))
# state_transformer = dec.StateTransformer.load(model_xml)
#
# dst_state = state_transformer.process(model3d.default_state,dec.StateTransformDirection.Forward)
# src_state = state_transformer.process(dst_state,dec.StateTransformDirection.Backward)
# print(len(dst_state))
# for d in range(len(src_state)):
#     print(model3d.default_state[d] - src_state[d],)


def dummy_transform(input_vec):
    print(' will it work?')
    return input_vec


state_trs = dec.StateTransformer()
state_trs.setStateTransformFunction(dummy_transform, 5,5)
input_vec = core.DoubleVector([1,2,3,4,5])
output_vec = state_trs.process(input_vec,dec.StateTransformDirection.Forward)
print(output_vec)


