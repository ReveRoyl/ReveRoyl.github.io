---
title: Pandas
top: false
cover: false
toc: true
mathjax: true
date: 2022-08-22 14:37
password: e3e831cb38631276253e39fdf3fe706a022cbf4b49e9282f878b2288f214c980
summary:
tags:
- Pandas
categories:
- programming
---

The objective is to calculate power spectrum for each frequencies and output data to excel files.

First load packages:

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
Then calculate power spectrum for each subject and frequency:
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
Last, output calculated power spectrum as xmlx files:
```python
df1 = pd.DataFrame({'sub1':abs_1,
                        'sub2':abs_2,
                        'sub3':abs_3,
                       'sub4':abs_4,
                       'sub5':abs_5,
                       'sub7':abs_7,
                       'sub8':abs_8,
                       'sub9':abs_9,
                       'sub10':abs_10,
                       'sub11':abs_11,
                       'sub13':abs_13,
                       'sub15':abs_15,
                       'sub16':abs_16,
                       'sub17':abs_17,
                       'sub18':abs_18,
                       'sub19':abs_19,
                       'sub20':abs_20,
                       'sub21':abs_21,
                      'sub22':abs_22,
                      'sub24':abs_24,
                      'sub25':abs_25,
                      'sub26':abs_26,
                      'sub27':abs_27,
                      'sub28':abs_28},)
df1.to_excel('models/abs.xlsx', sheet_name='sheet1', index=False)
```

```python
df2 = pd.DataFrame({'sub1':rel_1,
                        'sub2':rel_2,
                        'sub3':rel_3,
                       'sub4':rel_4,
                       'sub5':rel_5,
                       'sub7':rel_7,
                       'sub8':rel_8,
                       'sub9':rel_9,
                       'sub10':rel_10,
                       'sub11':rel_11,
                       'sub13':rel_13,
                       'sub15':rel_15,
                       'sub16':rel_16,
                       'sub17':rel_17,
                       'sub18':rel_18,
                       'sub19':rel_19,
                       'sub20':rel_20,
                       'sub21':rel_21,
                      'sub22':rel_22,
                      'sub24':rel_24,
                      'sub25':rel_25,
                      'sub26':rel_26,
                      'sub27':rel_27,
                      'sub28':rel_28},)
df2.to_excel('models/rel.xlsx', sheet_name='sheet1', index=False)
```

Hint: potential promotions are:

1. to use wavelet instead of Fourier transformation;
2. to use loop somehow pass data to DataFrame instead of listing them.
