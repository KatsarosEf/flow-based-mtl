import glob
import os
import shutil
import numpy
import cv2

src = '/media/efklidis/4TB/dblab_ecai/*/*'
seqs = glob.glob(src)

for seq in seqs:


    original = sorted(glob.glob(os.path.join(seq, 'GT', '*')), key=lambda x: int(os.path.basename(x)[:-4]))
    processed = sorted(glob.glob(os.path.join(seq, 'GTpro', '*')), key=lambda x: int(os.path.basename(x)[:-4]))

    out = cv2.VideoWriter(os.path.join(seq, 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1660, 800))
    for (o, p) in zip(original, processed):
        o_np = cv2.imread(o)
        p_np = cv2.imread(p)
        white = 255 * numpy.ones((800, 60, 3), dtype='uint8')
        stack = numpy.concatenate([o_np, white, p_np], axis=1)
        out.write(stack)
    out.release()