import os
import glob
import shutil
import pandas as pd

src = '/media/efklidis/4TB/dblab_real'
dst = '/media/efklidis/4TB/dblab_ecai_temp'

file = pd.read_csv('/home/efklidis/Downloads/DBREAL table - Sheet1.csv')

src_seqs = [x for x in glob.glob(os.path.join(src, '*', '*')) if 'train' in x or 'test' in x or 'val' in x]

for seq in src_seqs:
    dst_seq = seq.replace('dblab_real', 'dblab_ecai_temp')
    try:
        os.makedirs(dst_seq)
    except:
        continue
    os.makedirs(dst_seq + '/GT', exist_ok=True)
    os.makedirs(dst_seq + '/input', exist_ok=True)
    os.makedirs(dst_seq + '/masks', exist_ok=True)

    print(seq)
    seq_name = os.path.split(seq)[-1]
    row = file.loc[file[file.columns[0]] == seq_name]

    start_frame = row.values[0][4]
    start_frame_name = str(start_frame) + '.jpg'
    start_full_frame_name = os.path.join(seq, 'input', start_frame_name)

    list_full_frame_name = sorted(glob.glob(os.path.join(seq, 'input', '*')),
                                  key=lambda x: int(os.path.basename(x)[:-4]))
    list_full_gt_frame_name = sorted(glob.glob(os.path.join(seq, 'GT', '*')),
                                  key=lambda x: int(os.path.basename(x)[:-4]))
    list_full_mask_frame_name = sorted(glob.glob(os.path.join(seq, 'masks', '*')),
                                  key=lambda x: int(os.path.basename(x)[:-4]))

    start_idx = [idx for idx, name in enumerate(list_full_frame_name) if name == start_full_frame_name][0]
    end_idx =  start_idx + 400

    input_frames_to_move = list_full_frame_name[start_idx:end_idx]
    input_frames_to_move = [name for idx, name in enumerate(input_frames_to_move) if idx % 2 == 0]
    [shutil.copyfile(file, file.replace('dblab_real', 'dblab_ecai_temp')) for file in input_frames_to_move]

    gt_frames_to_move = list_full_gt_frame_name[start_idx:end_idx]
    gt_frames_to_move = [name for idx, name in enumerate(gt_frames_to_move) if idx % 2 == 0]
    [shutil.copyfile(file[:-4] + '.png', file.replace('dblab_real', 'dblab_ecai_temp')[:-4] + '.png') for file in gt_frames_to_move]

    mask_frames_to_move = list_full_mask_frame_name[start_idx:end_idx]
    mask_frames_to_move = [name for idx, name in enumerate(mask_frames_to_move) if idx % 2 == 0]
    [shutil.copyfile(file[:-4] + '.png', file.replace('dblab_real', 'dblab_ecai_temp')[:-4] + '.png') for file in mask_frames_to_move]








