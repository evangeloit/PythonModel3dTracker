import csv

# Converts the framerate of bvh files to match that of kinect sequence using a timestamp correspondences file.
# The output files should be converted (e.g. using Emacs) to have unix style new-line.

dims = 94
input_dir = "/data/local_home/mad/Development/Projects/htrgbd_scripts/ds/human_tracking/mhad/landmark_detections/gt_bvh/"
bvh_tmpl = "skl_s{0:02d}_a{1:02d}_r01.bvh"
output_bvh_tmpl = "skl_s{0:02d}_a{1:02d}_r01_sync.bvh"
crp_tmpl = "corr_moc_kin01_s{0:02d}_a{1:02d}_r01.txt"
subjects = [9]*11
actions = range(1,12)
# subjects = range(1,13)
# actions = [4]*12

for subject,action in zip(subjects,actions):
    input_bvh = input_dir + bvh_tmpl.format(subject,action)
    input_crp  = input_dir + crp_tmpl.format(subject,action)
    output_bvh = input_dir + output_bvh_tmpl.format(subject,action)

    print(subject,action)
    print(input_crp)
    print(input_bvh)
    print(output_bvh, '\n')

    frame_crp = []
    with open(input_crp) as crp:
        reader = csv.reader(crp, delimiter=' ')
        for r in reader:
            frame_crp.append(int(r[2]))
    print(frame_crp)
    bvh_frames_output = len(frame_crp)

    found_motion = False
    bvh_o = open(output_bvh,'w')
    writer = csv.writer(bvh_o, delimiter=' ')
    with open(input_bvh) as bvh:
        reader = csv.reader(bvh, delimiter=' ')
        bvh_frame_counter = 0
        for i,r in enumerate(reader):
            if len(r) == dims and found_motion:
                if bvh_frame_counter in frame_crp:
                    writer.writerow(r)
                bvh_frame_counter += 1
            else:
                if r[0] == 'MOTION': found_motion = True
                if r[0] == 'Frame': r[2] = str(1./30.)
                if r[0] == 'Frames:': r[1] = str(bvh_frames_output)
                writer.writerow(r)

