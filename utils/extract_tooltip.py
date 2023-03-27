import glob
import shutil
import cv2
import numpy as np
import os

seqs_dental = "/media/efklidis/4TB/dblab_ecai/*/*/GTpro/"
seqs = glob.glob(seqs_dental)


n = 100
for seq in seqs:
    frames_names = sorted(glob.glob(os.path.join(seq, '*')), key=lambda x: int(os.path.basename(x)[:-4]))
    # grab first n frames
    frames = [cv2.imread(frame) for frame in frames_names][:n][::10]
    # get combinations of all possible absolute differences
    frame_diff_1 = np.abs((frames[0] - frames[1]))
    frame_diff_2 = np.abs((frames[0] - frames[2]))
    frame_diff_3 = np.abs((frames[0] - frames[3]))
    frame_diff_4 = np.abs((frames[0] - frames[4]))
    frame_diff_5 = np.abs((frames[0] - frames[5]))
    frame_diff_6 = np.abs((frames[0] - frames[6]))
    frame_diff_7 = np.abs((frames[0] - frames[7]))

    average = np.mean((frame_diff_1, frame_diff_2,
                       frame_diff_3, frame_diff_4,
                       frame_diff_5, frame_diff_6,
                       frame_diff_7), 0)

    average = average - average.min()
    average = average / average.max()
    average = average[average<0.1]
    cv2.imwrite('./avg.png', average*255.0)

