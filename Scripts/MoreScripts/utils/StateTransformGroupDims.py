import os

import BlenderMBVLib.GenerateStateTransforms as gst
import PyMBVParticleFilter as pf

os.chdir(os.environ['bmbv']+'/Scripts/')


model_xml = Paths.model3d_dict['mh_body_male_meta'][1]
model3d_dst = pf.Model3dMeta.create(model_xml)
for d in range(model3d_dst.n_dims):
    print(d, model3d_dst.dim_names[d],model3d_dst.dim_types[d], model3d_dst.dim_axes[d], \
        model3d_dst.low_bounds[d], model3d_dst.default_state[d], model3d_dst.high_bounds[d])

global_scale_group = [i for i,t in enumerate(model3d_dst.dim_types) if t == pf.DimType.Scale ]
#global_scale_group = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.Scale
#                        and (a == pf.DimAxis.X or a == pf.DimAxis.Z)]
# scale_groups = []
# palm_scalewidth_group = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.Scale
#                     and ('palm' in n or 'hand' in n) and (a == pf.DimAxis.X or a == pf.DimAxis.Z)]
# palm_scalelength_group = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.Scale
#                     and ('palm' in n or 'hand' in n) and (a == pf.DimAxis.Y)]
# scale_groups.append(palm_scalelength_group)
# scale_groups.append(palm_scalewidth_group)
# for f in ['f_pinky','f_middle','f_ring','f_index','thumb']:
#     f_scalewidth_group = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.Scale
#                     and (f in n) and (a == pf.DimAxis.X or a == pf.DimAxis.Z)]
#     f_scalelength_group = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.Scale
#                     and (f in n) and (a == pf.DimAxis.Y)]
#     scale_groups.append(f_scalelength_group)
#     scale_groups.append(f_scalewidth_group)
#
#for g in scale_groups:
#    print g
#
spine_group_x = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.RotationEuler
                     and ('spine' in n or 'chest' in n) and (a == pf.DimAxis.X)]
spine_group_y = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.RotationEuler
                     and ('spine' in n or 'chest' in n) and (a == pf.DimAxis.Y)]
spine_group_z = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.RotationEuler
                     and ('spine' in n or 'chest' in n) and (a == pf.DimAxis.Z)]
right_shoulder_z = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.RotationEuler
                     and ('clavicle.R' in n or 'deltoid.R' in n or 'upper_arm.R' in n) and (a == pf.DimAxis.Z)]
left_shoulder_z = [i for i,(n,t,a) in enumerate(zip(model3d_dst.dim_names,model3d_dst.dim_types,model3d_dst.dim_axes)) if t == pf.DimType.RotationEuler
                     and ('clavicle.L' in n or 'deltoid.L' in n or 'upper_arm.L' in n) and (a == pf.DimAxis.Z)]



groups = [global_scale_group, spine_group_x, spine_group_y, spine_group_z, right_shoulder_z,left_shoulder_z]
model3d_src = gst.GenerateStateTransform_GroupNormalize(model3d_dst,groups)
model3d_src.model_name = model3d_dst.model_name + '_glbscl'
model3d_src.save()
