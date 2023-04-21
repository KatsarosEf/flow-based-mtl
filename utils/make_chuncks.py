import numpy as np
import glob
import cv2
import os

seqs_dental = "/media/efklidis/4TB/dblab_ecai/*/*/OF_backw/"
seqs = glob.glob(seqs_dental)


def estimate_homography_from_flo(flow, method):
    u, v = flow[:, :, 0], flow[:, :, 1]

    h, w = u.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x, y)
    dest_xx = xx + u
    dest_yy = yy + v

    xx = np.expand_dims(xx, axis=2).astype(np.float32)
    yy = np.expand_dims(yy, axis=2).astype(np.float32)
    dest_xx = np.expand_dims(dest_xx, axis=2).astype(np.float32)
    dest_yy = np.expand_dims(dest_yy, axis=2).astype(np.float32)


    if method=='4df':
        src_pts = np.concatenate((xx, yy), axis=2).reshape(-1, 2)
        dst_pts = np.concatenate((dest_xx, dest_yy), axis=2).reshape(-1, 2)
        H, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        H = np.vstack((H, np.array([[0, 0, 1]])))
    elif method=='8df':
        src_pts = np.concatenate((xx, yy), axis=2).reshape(-1, 2)
        dst_pts = np.concatenate((dest_xx, dest_yy), axis=2).reshape(-1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts)

    return H

M = 200
N = 200
# for seq in seqs:
#     flow_names = sorted(glob.glob(os.path.join(seq, '*')), key=lambda x: int(os.path.basename(x)[:-4]))
#
#     for flow_name in flow_names:
#         flow = np.load(flow_name)
#         tiles = [flow[x:x+M,y:y+N] for x in range(0,flow.shape[0],M) for y in range(0,flow.shape[1],N)]
#         homos = [estimate_homography_from_flo(tile, method='4df') for tile in tiles]


flow_names = [
    '/home/efklidis/ecai-dental/00021.npy',
    '/home/efklidis/ecai-dental/00023.npy',
    '/home/efklidis/ecai-dental/dentalflows/41070.npy',  # fe78c8b8-765a-44da-a10f-09a0651f3a6b
    '/home/efklidis/ecai-dental/dentalflows/85.npy',     # a847cec6-5bac-4148-a717-7cc5f20d1cd6
    '/home/efklidis/ecai-dental/dentalflows/21922.npy',  # 1f3041c2-2971-49eb-95e0-ced6f46e7b6e
    '/home/efklidis/ecai-dental/dentalflows/5536.npy',   # d8ac7e04-8e79-4f54-8f12-ee8d0ec08628
    '/home/efklidis/ecai-dental/dentalflows/51.npy',     # 0c5367f2-b43c-4af3-9751-0219c48796e9
    '/home/efklidis/ecai-dental/dentalflows/24791.npy',  # a1dfdc70-6320-4ab8-8087-cec8adf22546
    '/home/efklidis/ecai-dental/dentalflows/29958.npy',  # d32b2069-4c61-4bdd-8857-ccf368695c85
    '/home/efklidis/ecai-dental/dentalflows/1294.npy',   # d9715e62-092c-4517-ba57-8b7eed6e18a9
    '/home/efklidis/ecai-dental/dentalflows/16846.npy',  # f821e60c-62d2-485c-8720-ad9c8530ba34
    '/home/efklidis/ecai-dental/dentalflows/668.npy',    # fbbba64e-127d-4eb1-a0d3-77af1baa230e

    '/home/efklidis/ecai-dental/dentalflows/5866.npy',   # f921a4fe-bbe2-4f29-90e7-40c024363840

]
np.set_printoptions(suppress=True)

for flow_name in flow_names:

    print(flow_name)
    if flow_name.split('/')[-1][:-4].startswith('0'):
        flow = np.transpose(np.load(flow_name)[0])
        # print(flow.shape)
        print('DAVIS')
    else:
        flow = np.load(flow_name)
        # print(flow.shape)
        # print('DENTAL')


    tiles = [flow[x:x+M,y:y+N] for x in range(0,flow.shape[0],M) for y in range(0, flow.shape[1], N)]
    # print("Length of tiles:{}".format(len(tiles)))
    homos = [estimate_homography_from_flo(tile, method='8df') for tile in tiles]
    stacked_homos = np.stack((homos))

    vars = np.round(np.var((stacked_homos), 0), 9)
    means = np.round(np.mean((stacked_homos), 0), 9)

    print("Averages:")
    print(means)
    print("Variances:")
    print(vars)
