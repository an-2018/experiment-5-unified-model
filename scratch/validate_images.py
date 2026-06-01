import cv2
import glob
import numpy as np
import os

imgs = glob.glob('artifacts/figures/phase_02_preprocessing/*.png') + glob.glob('artifacts/figures/phase_03_unimodal_baselines/*.png')
results = []
for img in imgs:
    i = cv2.imread(img)
    if i is not None:
        unique = len(np.unique(i.reshape(-1, i.shape[2]), axis=0))
        h, w = i.shape[:2]
        is_empty = unique < 1000 # Just an arbitrary heuristic
        results.append(f"{os.path.basename(img)}: {w}x{h}, unique_colors={unique}")
    else:
        results.append(f"{os.path.basename(img)}: Failed to load")

print('\n'.join(results))
