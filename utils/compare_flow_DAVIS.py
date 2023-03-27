import glob
import shutil
import numpy as np
import os
import matplotlib.pyplot as plt

# src_davis = "/media/efklidis/4TB/DAVIS-2017-trainval-480p/DAVIS/Flows/*/"
# seqs = glob.glob(src_davis)
# all_flows = []
# for seq in seqs:
#     flows = glob.glob(os.path.join(seq, '*'))
#     seq_flows = np.concatenate([np.load(flow).astype(np.float16) for flow in flows])
#     seq_flows = np.sqrt(seq_flows[:,0,:,:,]**2 + seq_flows[:,1,:,:,]**2).flatten()
#     all_flows.append(seq_flows)
#     print(len(all_flows))
#
#
# davis_flows = np.concatenate(all_flows)
# np.save('./davis_dist.npy', davis_flows)

# seqs_dental = "/media/efklidis/4TB/dblab_ecai/*/*/OF_backw/"
# seqs = glob.glob(seqs_dental)
# all_flows = []
# for seq in seqs:
#     flows = glob.glob(os.path.join(seq, '*'))
#     seq_flows = np.array([np.load(flow).astype(np.float16) for flow in flows])
#     seq_flows = np.sqrt(seq_flows[:, :, :, 0] ** 2 + seq_flows[:, :, :, 1] ** 2).flatten()
#     all_flows.append(seq_flows)
#     print(len(all_flows))
#
#
# dental_flows = np.concatenate(all_flows)
# np.save('./dental_dist.npy', dental_flows)

dental_flows = np.load('./dental_dist.npy')
davis_flows = np.load('./davis_dist.npy')

plt.figure(figsize=(8,6))
plt.hist(dental_flows[dental_flows<50], bins=80, density=True, alpha=0.5, label="Dental Data")
plt.xlabel("Pixel Displacements", size=18)
plt.ylabel("Density", size=18)
plt.title("Motion Magnitudes", size=18)
plt.legend(loc='upper right')
plt.savefig("dental.png")





# plt.figure(figsize=(8,6))
# plt.hist(dental_flows[dental_flows<50], bins=80, density=True, alpha=0.5, label="Dental Data")
# plt.hist(davis_flows[davis_flows<50], bins=80, density=True, alpha=0.5, label="Davis Data")
# plt.xlabel("Pixel Displacements", size=18)
# plt.ylabel("Density", size=18)
# plt.title("Motion Magnitudes", size=18)
# plt.legend(loc='upper right')
# plt.savefig("overlapping_histograms_with_matplotlib_Python_2_distancessss_density.png")





