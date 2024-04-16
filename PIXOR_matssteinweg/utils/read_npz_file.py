import numpy as np

data = np.load('/media/lucasrdalcol/data/phd_research/results/3d-object-detection-experiments/PIXOR_matssteinweg/metrics/history.npz', allow_pickle=True)
lst = data.files
for item in lst:
    print(item)
    print(data[item])