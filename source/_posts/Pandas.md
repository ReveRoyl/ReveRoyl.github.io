---
title: Cpp Learning
top: false
cover: false
toc: true
mathjax: true
date: 2022-08-22 14:37
password:
summary:
tags:
- Pandas
categories:
- programming
---

```python
import os
import sys
sys.path.append('Code/code/')
from load_data import load_MEG_dataset
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.cuda import amp
import torch.nn.functional as F
import math
from scipy.integrate import simps
from mne.time_frequency import psd_array_welch
from band_power import (
    bandpower_multi_bands,
    standard_scaling_sklearn,
)
import sklearn
import pandas as pd

```

```python
# for sub in [sub for sub in range (1,29) if sub not in [6, 12, 14 ,23]]:
for sub in range(1,2):
    exec('abs_{} = []'.format(sub))
    exec('rel_{} = []'.format(sub))
    Split = 0.90
    X_train, y_train = load_MEG_dataset([str(i).zfill(3) for i in range(sub,sub+1)], mode = 'concatenate', output_format='numpy',shuffle = False, training=True, train_test_split=Split, batch_size=500)#, pca_n_components =30)
    X_test, y_test = load_MEG_dataset([str(i).zfill(3) for i in range(sub,sub+1)], mode = 'concatenate', output_format='numpy',shuffle = False, training=False, train_test_split=Split, batch_size=500)#, pca_n_components =30)

    X_train, X_test = (X_train-X_train.mean())/X_train.std(), (X_test-X_test.mean())/X_test.std()

    X_train = X_train[:, None, ...]
    X_test = X_test[:, None, ...]

    y_train = (y_train / 2) - 1
    y_test = (y_test / 2) - 1

    X = np.swapaxes(X_train, 2, -1).squeeze()
    data = X[X.shape[0]-1, 70, :]
    psd_mne, freqs_mne = psd_array_welch(data, 100, 1., 70., n_per_seg=None,
                            n_overlap=0, n_jobs=1)
    for low, high in [(0.5, 4), (4, 8), (8, 10), (10, 12), (12, 30),
                    (30, 70)]:
        print("processing bands (low, high) : ({},{})".format(low, high))
        # Find intersecting values in frequency vector
        idx_delta = np.logical_and(freqs_mne >= low, freqs_mne <= high)
        # Frequency resolution
        freq_res = freqs_mne[1] - freqs_mne[0]  # = 1 / 4 = 0.25

        # Compute the absolute power by approximating the area under the curve
        power = simps(psd_mne[idx_delta], dx=freq_res)
        exec('abs_{}.append(power)'.format(sub))
        # print('Absolute power: {:.4f} uV^2'.format(power))
        
        total_power = simps(psd_mne, dx=freq_res)
        rel_power = power / total_power
        exec('rel_{}.append(rel_power)'.format(sub))
        # print('Relative power: {:.4f}'.format(rel_power))
```

```
```

