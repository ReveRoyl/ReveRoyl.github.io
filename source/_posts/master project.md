---
title: master project
top: false
cover: false
toc: true
mathjax: true
date: 2022-05-16 14:00
password:
summary:
tags:
- project
categories:
- Neruroscience programming
---

# April 12th

## First meeting 

### Objective: 

to find out which data we are going to use and which method to analyse it

### Results:

data: https://openneuro.org/datasets/ds003682

The file we cares:

![image-20220411140935965](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204111409061.png)

We can use MNE to open it

What these files do:

https://mne.tools/stable/generated/mne.read_epochs.html

More templates can be found in:

https://github.com/tobywise/aversive_state_reactivation/blob/master/notebooks/templates/sequenceness_classifier_template.ipynb

# April 13th

## First session meeting 

### Objective:

learn how to write a lab notebook

![image-20220413104227189](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131042297.png)

![image-20220413104255487](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131042577.png)

![image-20220413104725413](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131047482.png)

![image-20220413105233836](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131052901.png)

### Results

General understanding:

Title: the utility of multi-task machine learning for decoding brain states

Table of Contents:

page numbers; date; title/subject/experiment

Gantt charts is good to help organise time:

![image-20220413110103327](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131101425.png)



# April 18th

## Install MNE-Python via pip:

### Objective:

to upgrade environment manager and compiler to the latest stable version

### Results:

Update Anaconda via ` conda upgrade --all` and  `conda install anaconda=2021.10`

The Python version I am using is:

```
Python 3.9.12	
```

get the data:

`git clone https://github.com/OpenNeuroDatasets/ds003682`

Install MNE:

`conda install --channel=conda-forge mne-base`

# April 19th

## Second meeting 

### Objective: 

have a general understanding of the data and figure out details about methods

`x_raw`

`x_raw.shape`

`y_raw`

`time = localiser epchoes`

### Results:

#### what I have known:

**scikit-learn** is the package I am going to use

use **PCA** to reduce dimensions 

![image-20220419104136746](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204191041843.png)

#### what need to be learned: 

##### 1. normalization, regularization

lasso L1 = sets unpredictive features to 0

ridge L2 = minimises the weights on unpredictive features

elastic net L1/L2

##### 2. search

`random_search randomizedsearchCV` to test the performance

##### 3. Neural network

neural network can be the best way for logistic leaning 

##### 4. validation	

# April 24th

### Objective:

play with existing code and make a foundation for the following coding

### Results:

#### Epochs

Extract signals from continuous EEG signals for specific time windows, which can be called epochs.

Because EEGs are collected continuously, to analyse EEG event-related potentials requires "slicing" the signal into time segments that are locked to time segments within an event (e.g., a stimulus).

#### Data corruption

The MEG data in the article is invalid, possibly because of data corruption

**incomplete copying led to corrupted files**

The following events are present in the data: 1, 2, 3, 4, 5, 32

```
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
'Visual/Left': 3, 'Visual/Right': 4,
'smiley': 5, 'button': 32}
```

`sklearn.cross_validation` has been deprecated since version 1.9, and `sklearn.model_selection` can be used after version 1.9.

# April 25th

## Install scikit-learn (sklearn)

use the command line 

`conda install -c anaconda scikit-learn`

Start the base environment in Anaconda Prompt: `activate base`

And install jupyter notebook, numpy and other modules in the environment

```
conda insatll tensorflow

conda install jupyter notebook

conda install scikit-learn

conda install scipy
```

how to choose the right algorithm 

https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

![flow chart of scikit](https://scikit-learn.org/stable/_static/ml_map.png)

## Learning sklearn

1. import modules

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```

2. create data

load `iris` data，store the **attributes** in  `X`，store the **labels** in `y`：

```
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
```

Looking at the dataset, `X` has four attributes, and `y` has three categories: 0, 1, and 2:

```
print(iris_X[:2, :])
print(iris_y)

"""
Output:
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 """
```

Divide the data set into training set and test set, where `test_size=0.3`, that is, the test set accounts for 30% of the total data:

```
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)
```

It can be seen that the separated data sets are also disrupted in order, which is better to train the model:

```
print(y_train)

"""
Outputs:
[2 1 0 1 0 0 1 1 1 1 0 0 1 2 1 1 1 0 2 2 1 1 1 1 0 2 2 0 2 2 2 2 2 0 1 2 2
 2 2 2 2 0 1 2 2 1 1 1 0 0 1 2 0 1 0 1 0 1 2 2 0 1 2 2 2 1 1 1 1 2 2 2 1 0
 1 1 0 0 0 2 0 1 0 0 1 2 0 2 2 0 0 2 2 2 1 2 0 0 2 1 2 0 0 1 2]
 """
```

3. Build a model - train - predict

Define the module method `KNeighborsClassifier()`, use `fit` to train `training data`, this step completes all the steps of training, the latter `knn` is already a trained model, which can be used directly `predict` For the data of the test set, comparing the value predicted by the model with the real value, we can see that the data is roughly simulated, but there is an error, and the prediction will not be completely correct.

```
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)

"""
[2 0 0 1 2 2 0 0 0 1 2 2 1 1 2 1 2 1 0 0 0 2 1 2 0 0 0 0 1 0 2 0 0 2 1 0 1
 0 0 1 0 1 2 0 1]
[2 0 0 1 2 1 0 0 0 1 2 2 1 1 2 1 2 1 0 0 0 2 1 2 0 0 0 0 1 0 2 0 0 2 1 0 1
 0 0 1 0 1 2 0 1]
 """
```

![image-20220425150724894](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204251507965.png)

## Succeed at drawing plot

```
import mne
import os
from mne.datasets import sample
import matplotlib.pyplot as plt

# The storage path of sample
data_path = sample.data_path()
# The storage path of the fif file
fname = 'E:\Proj\Previous data\sample\MEG\sample\sub-001_localiser_sub-001_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz'

epochs = mne.read_epochs(fname)

print(epochs.event_id)

picks = mne.pick_types(epochs.info, meg=True, ref_meg=False, exclude='bads')

epochs.plot(block=True)

epochs.plot_drop_log()

plt.show()
```

```
epochs = mne.read_epochs(fname)

evoked = epochs.average()
evoked.plot_topomap()

plt.show()
```
```
availabe_event = [1, 2, 3, 4, 5, 32]

for i in availabe_event:
    evoked_i = epochs[i].average(picks=picks)
    epochs_i = epochs[i]
    evoked_i.plot(time_unit='s')
    plt.show()

```

## A few questions at last: 

1. why do we split the data as 70%, does it work as other ration?

   Because we have got enough data to train and need more data to test and valid the training performance.

# April 26th

## MRI safety training for 2.5 hrs

## update Anaconda (start to use a new platform)

`conda update conda`

`conda update anaconda`

`conda update --all`

done

Python version: 3.8.13-h6244533_0

## change the directory path of Jupyter notebook

1. `jupyter notebook --generate-config` get the config file. change the line `c.NotebookApp.notebook_dir = ''` to the directory I want

2. find jupyter notebook file, change the attributes. ![change the attributes](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204261614733.png)

## Link the local directory and Github

for the convenience of collaboration 

SSH connect public key (id_rsa.pub) was created before.

after create the directory, run the command in git:

```
git init
git add .
git git commit -m  "Comment"
git remote add origin "the url of directory"
git push -u origin main
```

## Journal club preparation for the next session

# April 27th

## Pycharm (use IDE)

Get Pycharm educational version via King’s email

install python 3.8 environment for running the code from https://github.com/tobywise/aversive_state_reactivation (because it was coding with python 3.8 compiler)

run below code in pycharm

```
!conda create -n py38 python=3.8
!pip install mne
!pip install scikit-learn
!pip install plotly
!pip install cufflinks
!pip install networkx
!conda install numba
!pip install pyyaml
!pip install papermill
```

## Fixation for some expired code

The function `joblib` does not exist in `sklearn.external` anymore.

Error occurs when run the function plot_confusion_matrix:

Deprecated since version 1.0: `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the following class methods: `from_predictions` or `from_estimator`.

use

```
ConfusionMatrixDisplay.from_predictions(y, y_pred)
```

instead of 

```
plot_confusion_matrix(mean_conf_mat[:n_stim, :n_stim], title='Normalised confusion matrix, accuracy = {0}'.format(np.round(mean_accuracy, 2)))
```

## Second session meeting

Every people gives a general introduction of their project

# April 28th

## Logistic regression cost function

The function is using the principle of maximum likelihood estimation to find the parameters $\theta$ for different models. At the meantime, a nice property is it is convex. So, this cost function is generally everyone use for fitting parameters in logistic regression.![image-20220429010526272](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204290105356.png)

The way we are going to minimize the cost function is using gradient descent:

![image-20220429011210521](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204290112574.png)

Other alternative optimization algorithms (no need to manually pick $\alpha$ studying rate):

1. Conjugate gradient
2. BFGS
3. L-BFGS

## Methods to storage trained model

set and train a simple SVC model

```
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)
```
Storage:

1. pickle

```
import pickle #pickle module

#store Model(Note: The save folder must be created in advance, otherwise an error will be reported)
with open('save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

#load Model
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    #test loaded Model
    print(clf2.predict(X[0:1]))
```

2. joblib (supposed to be faster when dealing with a large data, because the use of multiprocessing)

```
from sklearn.externals import joblib #jbolib模块

#store Model(Note: The save folder must be created in advance, otherwise an error will be reported)
joblib.dump(clf, 'save/clf.pkl')

##load Model
clf3 = joblib.load('save/clf.pkl')

#test loaded Model
print(clf3.predict(X[0:1]))

```

# April 29th

## Google Colab

Get the subscription of Google Colab and Google drive

clone data to google drive

Token: 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

connect Google Colab with VS code

## Regularization

We can use **regularization** to rescue the overfitting

There are two types of regularization:

- **L1 Regularization** (or Lasso Regularization)

  $Min$($$\sum_{i=1}^{n}{|y_i-w_ix_i|+p\sum_{i=1}^{n}|w_i|}$$)

- **L2 Regularization** (or Ridge Regularization)

  $Min$($$\sum_{i=1}^{n}{(y_i-w_ix_i)^2+p\sum_{i=1}^{n}w_i^2}$$)

where `p` is the tuning parameter which decides in what extent we want to penalize the model.

However, there is another method for combination

- **Elastic Net:** When L1 and L2 regularization combine together, it becomes the elastic net method, it adds a hyperparameter.

how to select:

| **S.No** | **L1 Regularization**                                   | **L2 Regularization**                                        |
| -------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **1**    | Panelizes the sum of absolute value of weights.         | penalizes the sum of square weights.                         |
| **2**    | It has a sparse solution.                               | It has a non-sparse solution.                                |
| **3**    | It gives multiple solutions.                            | It has only one solution.                                    |
| **4**    | Constructed in feature selection.                       | No feature selection.                                        |
| **5**    | Robust to outliers.                                     | Not robust to outliers.                                      |
| **6**    | It generates simple and interpretable models.           | It gives more accurate predictions when the output variable is the function of whole input variables. |
| **7**    | Unable to learn complex data patterns.                  | Able to learn complex data patterns.                         |
| **8**    | Computationally inefficient over non-sparse conditions. | Computationally efficient because of having analytical solutions. |

### Question:

which layer should I apply the regularization?

From the model's summary, I can determine which layers have the most parameters. It is better to apply regularization to the layers with the highest parameters.

In the above case, how to get each layer’s parameters?

```python
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model)
```

# May 2nd

## Question ahead:

get a question: how to determine the classifier centre? 

for this case, it is around 20 to be at the middle/ top

GPU accelerations

It could be a little bit tricky to accelerate the calculation in sklearn with GPU. Here is a possible solution: https://developer.nvidia.com/blog/scikit-learn-tutorial-beginners-guide-to-gpu-accelerating-ml-pipelines/.  

## Third meeting:

possible deep learning package:

JAX, HAIKU

my aim is to inform bad-performance data with the training model of good-performance data in aims to increase the performance. One hallmark is to increase the mean accuracy of each cases as high as possible.

# May 3rd

Successfully run the code for confusion matrix.

pictures of each case are stored in the Github depository: 

https://github.com/ReveRoyl/MT_ML_Decoding/tree/main/Aversive_state_reactivation/notebooks/templates/save_folder

It takes around 36 minutes to run 28 cases. 

Mean accuracy with existing code:

```
[0.4288888888888889, 0.33666666666666667, 0.2777777777777778, 0.5022222222222222, 0.5066666666666667, 0.4245810055865922, 0.5577777777777778, 0.43222222222222223, 0.65, 0.47888888888888886, 0.3377777777777778, 0.4800469483568075, 0.27111111111111114, 0.37193763919821826, 0.4288888888888889, 0.40555555555555556, 0.46444444444444444, 0.7077777777777777, 0.5811111111111111, 0.4711111111111111, 0.4255555555555556, 0.5022222222222222, 0.45394006659267483, 0.38555555555555554, 0.6222222222222222, 0.4622222222222222, 0.35444444444444445, 0.47444444444444445]
```
# May 10th

Get rid of the effect of null data

transform X into the same size as clf

# May 11th

## Grid Search CV

Test Grid Search CV instead of Randomized Search CV

However, the result of grid search CV is worse than the randomized search CV. I think the reason is that random search CV uses a random combination of hyperparameters to find the best solution for a built model. However, the random search CV is not 100% better than grid search CV: the disadvantage of random search is that it produces high variance during computation.

## Concatenation

### Objective

Since my aim is to transfer the model prediction of one case to anther, I try to concatenate multiple cases data together to train the model and see what will happen.

### Result

Concatenate X np arrays and test, the mean accuracy increases. I test 5 cases.

Mean accuracy = 0.5111111111111111, while for each cases, previous mean accuracy = 0.43777777777777777; 0.34; 0.2788888888888889; 0.5055555555555555.

![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

The concatenated data gives a better fit:

![output](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202205121731546.png)

However, the generalization ability of this method has not been decided yet till now.

# May 12th

add ‘participant number’ dimension to X

```python
X_4D = []
for session_id_int in range(1, 2):
    session_id = '{:03d}'.format(session_id_int)
    print(session_id)
    # Get data
    localiser_epochs = mne.read_epochs(os.path.join(output_dir, 'preprocessing', 'sub-{}', 'localiser', 'sub-{}_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz').format(session_id,session_id))  
    
    X_raw = localiser_epochs.get_data()
    
    picks_meg = mne.pick_types(localiser_epochs.info, meg=True, ref_meg=False)
    event_selector = (y_raw < n_stim * 2 + 1)
    X_raw = X_raw[event_selector, ...]
    X_raw = X_raw[:, picks_meg, :]
    X = X_raw.copy()
    X = X[..., classifier_center_idx + classifier_window[0]:classifier_center_idx + classifier_window[1]] 

    X_4D.append(X)
# for epoch in localiser_epochs_concatenate:
#     print(epoch.shape)
# localiser_epochs = mne.concatenate_epochs(localiser_epochs_concatenate)
X = np.stack(X_4D)
```

It takes me 1 hour to make the code elegant.

# May 13th

## Forth meeting

### Objective:

We have tested existing code, the following step is to try different models to predict.

My supervisor’s suggestion is to use haiku trying CNN.

```
pip install -upgrade jax optax dm-haiku 
```

to be learnt: 

active function: rectifier RELU

### Result:

New model selection: CNN with Haiku

# May 14th

Find a  bug: concatenated data was not giving a correct result. The better performance was because of the randomness of every time training. The concatenated data won’t give a better prediction. 

possible solution: transfer learning instead of directly concatenation.

# May 16th

### Objective

learning deep learning courses (Andrew Ng) in coursera

### Result:

In the context of artificial neural networks, the rectifier or ReLU (Rectified Linear Unit) activation function is an activation function defined as the positive part of its argument.

#### Convolution

Types of layer in a CNN:

- Covolution
- Pooling
- Fully connected 

Cases:

Classic networks: 

- LeNet-5
- AlexNet
- VGG

ResNet

Inception

LeNet -5 

# May 17th

## Learning JAX

**Question: why jax has its own numpy type: jax.numpy? what is the difference between it and numpy?**

Jax.numpy is a little bit different from numpy: the former is immutable

**Question: why do we need jax.numpy?** 

because numpy only works on CPU while jax.numpy works on GPU

**another advantage of JAX**

`jit()` can be used to compile the data input thus makes the program run faster

## First group meeting

Yiqi introduced his recent work which is based on U-net, VAE

# May 18th

## Learning JAX

```python
grad() # for differentation
from jax import jacfwd, jacrev # for Jacobian matrix
vmap() # for vectorization; it can makes you write your functions as if you were dealing wiht a single datapoint
```

JAX API structure

- NumPy <-> lax <-> XLA
- lax API is stricter and more powerful
- It's a Python wrapper around XLA

![nn](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202207221751704.svg)

```python
class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        # self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), padding="SAME")
        # self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), padding="SAME")
        self.conv1 = hk.Conv2D(output_channels=64, kernel_shape=(11,11), stride=4, padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=192, kernel_shape=(5,5), padding="SAME")
        self.conv3 = hk.Conv2D(output_channels=384, kernel_shape=(3,3), padding="SAME")
        self.conv4 = hk.Conv2D(output_channels=256, kernel_shape=(3,3), padding="SAME")
        self.conv5 = hk.Conv2D(output_channels=256, kernel_shape=(3,3), padding="SAME")
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(len(classes))

    def __call__(self, x_batch):
        x = self.conv1(x_batch)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(3, 3), strides=(2, 2), padding='VALID')(x)        
        x = self.conv3(x_batch)
        x = jax.nn.relu(x)
        x = self.conv4(x_batch)
        x = jax.nn.relu(x)
        x = self.conv5(x_batch)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(3, 3), strides=2, padding='VALID')(x)
        # x = hk.AvgPool(window_shape=(6, 6), strides=(2, 2), padding='SAME')(x)
        
        x = self.flatten(x)
        x = self.linear(x)
        x = jax.nn.softmax(x)
        return x
```

# May 19th

## Test CNN

CNN test was run but the prediction result is not as good as expected: 

the performance of it is about 0.4 for training dataset and 0.15 for test dataset.

I think we should try to tune the parameters of model or test a different model since the current model is suitable for image recognition. Before that, literature reviews should be done.

## Fifth meeting

I have the requirement of visualization during training: 

To use Tensor Board to show the training process

# May 20th

## Tried AlexNet (can be easily get from Pytorch hub)

AlexNet

```
AlexNet  = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).cuda()
```



Resnet 18

```
Resnet   = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).cuda()
```



![image-20220722175851266](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202207221758352.png)



## Sixth meeting

### Objective:

discuss about that if we should use concatenate data and pre-processing for standardization 

### Results:

Dr Toby and I had a discussion together with Dr Rossalin 

Rossalin advices us to try transfer learning or LSTM.

# May 24th

## Second group meeting

Zarc gives a presentation about his NLP project.

# May 27th

## Learning Tensorflow

### Session control

In Tensorflow, a string is defined as a variable, and it is a variable, which is different from Python. (later on I found it was correct in Tensorflow v1 but changes in Tensorflow v2)

 `state = tf.Variable()`

```
import tensorflow as tf

state = tf.Variable(0, name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)
```

If you set variables in Tensorflow, initializing the variables is the most important thing! ! So after defining the variable, be sure to define `init = tf.initialize_all_variables()` .

At this point, the variable is still not activated, and it needs to be added in `sess` , `sess.run(init)` , to activate`init`. (again, it changes in v2, we do not need to initialize the session in Tensorflow v2)

```Python
# Variable, initialize
# init = tf.initialize_all_variables() # expired
init = tf.global_variables_initializer()  
 
# Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

```

Note: directly `print(state)` does not work! !

Be sure to point the `sess` pointer to `state` and then `print` to get the desired result!

### placeholder

`placeholder` is a placeholder in Tensorflow that temporarily stores variables.

If Tensorflow wants to pass in data from the outside, it needs to use `tf.placeholder()`, and then transfer the data in this form `sess.run(***, feed_dict={input: **})`.

Examoles：

```
import tensorflow as tf

# Tensorflow requires defining placeholder's type, usually float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# multiply input1 and input2
ouput = tf.multiply(input1, input2)
```

Next, the work of passing the value is handed over to `sess.run()`, the value that needs to be passed in is placed in `feed_dict={}` and corresponds to each `input` one by one. `placeholder` and `feed_dict={ }` are bound together.

```
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]
```

### Activation function

when there are many layers, be careful to use activation function in case of the **gradient exploding or gradient disappearance**.

# May 28th

> Although in the computer vision field convolutional layer often followed by a pooling layer to reduce the data dimension at the expense of information loss, in the scenes of MEG decoding, the size of MEG data is much smaller than the computer vision field. So in order to keep all the information, we don’t use the pooling layer. After the spatial convolutional layer, we use two layers of temporal convolutional layers to extract temporal features, a fully connected layer with dropout operation for feature fusion, and a softmax layer for final classification. (Huang2019)

It is important to select if we should use pooling layers and how many to use.

# May 30th

## ML optimizer

### Objective:

learn to use and select different optimizer

### Results:

- Stochastic Gradient Descent (SGD)
- Momentum
- AdaGrad
- RMSProp
- Adam

![speedup3](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202205302233928.png)

X.shape gives (n_epochs, n_channels, n_times) corresponding to (batches, pixels,channels)  of images

### impression

As I tried to apply regularization at the same time. It is important to bear in mind, if we give a momentum in SGD (about 0.9), we can easily apply L2 regularization when the weight_decay is larger than 0.

# May 31st

### Objective:

try to learn data augmentation and learn a important rule: “No free lunch theorem”

### Results:

D. H. Wolpert et. al. (1995) come up with “No free lunch theorem”: all optimization algorithms perform equally well when their performance is averaged across all possible problems.

For any prediction function, if it performs well on some training samples, it must perform poorly on other training samples. If there are certain assumptions about the prior distribution of the data in the feature space, there are as many good and bad performances.

# Jun 1st

## Seventh meeting

Dr Toby provide me the code to load data with dataloader

# Jun 6th

### Objective:

try different loss function:

### Results:

when apply other loss function instead of CrossEntropy, there are some format error occurs:

solution: transform inputs from multi hot coding into one hot coding. 

```Python
def onehot(batches, n_classes, y):
  yn = torch.zeros(batches, n_classes)
  for i in range(batches):
    x = [0 for j in range(batches)]
    x[i] = y[i]/2-1                     #ex. [12]-> [5]
    yn[i][int(x[i])]+= 1                  #[000010000]
  return yn
```

Later I found that we can use the function in Pytorch: F.onehot

# Jun 7th

try different models, reshape the data from [batches, channels, times point ]to be [batches, times point, channels] in corresponding to images format [batches, picture channels (layers), pixels] (not the same channels, the same names but different meanings, former one is electrode channels. latter one is picture layers)

# Jun 8th

## Third session meeting

### Objective:

learn the key points of writing a good introduction

### Results:

from the history of ML to the current meaning 

- [x] The **history** of **ML**, (black box...)
- [ ] the **history** of disease classification or **behaviour prediction**
- [ ] how people use the ML techniques to **predict  behaviours** these days
- [ ] what is aversive state reactivation (one sentence); why people want to learn aversive state reactivation
- [ ] how people tried to connect ML with aversive state reactivation
- [ ] the problem (gap) is previous model only gives a low prediction accuracy
- [ ] Introduction of CNN, LSTM, RNN or transfer learning
- [ ] My aims is to optimize the model with new techniques: CNN, LSTM RNN or transfer learning

Note: state what in each paragraph is not enough and leads to the next paragraph.

## LSTM RNN

### Objective: 

try LSTM RNN

### Results:

The problems when I try to use one hot data `  labels = F.one_hot(labels)`, it is not applicable because in that case the dimensions could be 28 instead of 14 as we only have even numbers,

Some packages automatically figure out where there missing labels during one hot operation but this one (pytorch) doesn't

## Eighth meeting

### questions: 

- [ ] loss functions give similar values?
- [x] ~~is it correct to use enumerate to loop each data?~~

In my case, CrossEntropy is the better for MEG data. It is the experience gained from Dr Toby. Even though I found the others’ research about hands behaviour prediction used MSE loss. It is not suitable for our results because theirs is about the movements while ours is not.

# Jun 9th

## Label noise.

### Objectives:

labels format it self may affect the model performance. 

**question**: how to determine numbers of hidden layers in LSTM RNN.

### Results:

We cannot correctly compare each label when we are using **one-hot format**. If the labels you predict are apples, bananas, and strawberries, obviously they do not directly have a comparison relationship. If we use 1, 2, 3 as labels, there will be a comparison relationship between the labels. Distances are different. With the comparison relationship, the distance between the first label and the last label is too far, which affects the learning of the model.

One promotion:

> Knowledge distillation (KD) improves performance precisely by suppressing this label nosie. Based on this understanding, we introduce a particularly simple improvement method of knowledge disillation, which can significantly improve the performance of ordinary KD by setting different temperatures for each image. 
>
> Xu, K., Rui, L., Li, Y., Gu, L. (2020). *Feature Normalized Knowledge Distillation for Image Classification.* In: Vedaldi, A., Bischof, H., Brox, T., Frahm, JM. (eds) Computer Vision – ECCV 2020. ECCV 2020. Lecture Notes in Computer Science(), vol 12370. Springer, Cham. https://doi.org/10.1007/978-3-030-58595-2_40

**Hidden layers in LSTM RNN**

The effect of the number of hidden layers for neural networks

![img](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202206120108575.jpeg)

# Jun 10th

try `CrossEntropyLoss()` with onehot format data

convert the labels from “every 2 in 0 to 28” to “every 1 in 1 to 14”

use `  with torch.autocast('cuda')` to solve “cuda error” of `CrossEntropyLoss()`

# Jun 12th

even though the accuracy of prediction for train data increases quickly as trainning, the accutacy for test data is still very low. It seems the validation loss does not converge. I decide the next step is going to do more literatural research and adjust the parameters.

# Jun 13th

### Problems:

the problem is overfitting. There could be 2 alternative options: 1, get more data; 2, try different learning rate or dynamic learning rate.

the way the train-test split worked wasn't ideal in the data loading function - because the data wasn't shuffled prior to splitting, the train and test set would often consist of different subjects if we load multiple subjects. The data can be shuffled before splitting (by setting shuffle=True), which means that there will be a mix of subjects in both the training and testing data. This seems to boost accuracy in the test set a little bit.

# Jun 14th

### Objective:

try `CosineEmbeddingLoss()`

> Cosine loss could have better performance for small dataset (around 30%). 
>
> Barz, B., & Denzler, J. (2019). Deep Learning on Small Datasets without Pre-Training using Cosine Loss. *Proceedings - 2020 IEEE Winter Conference on Applications of Computer Vision, WACV 2020*, 1360–1369. https://doi.org/10.48550/arxiv.1901.09054

### Results:

however, the performance is not promoted

# Jun 15th

## Fourth session meeting

I read others work, extract the key words and main ideas:

> *Affective modulation of the startle response in depression: Influence of the severity of depression, anhedonia and anxiety*

startle reflex (SR)

1. affective rating

​	get affective rating after participants watched every clips 

​	and analysis the difference between depressed patients and control.

2. Startle amplitude

3. EMG

   higher baseline EMG activity during pleasant and unpleasant clips, relative to the neutral clips

**Conclusion**

a reduced degree of self-reported mood modulation 

**Gap**

The findings differ from those of Allen et al. (1999): they think it does not matter for depression or anhedonia for affective and emotional modulation.

**key wards**

Depression

Anxiety

Anhedonia

Affective modulation

Mood regulation

Startle response

EMG

affective rating

## Ninth meeting

### Problems:

Since the past work does not give a good result and it has been nearly halfway of the project. My following work is suggested to be focused on:

1. Multilayer Perceptron

2. transfer learning

# Jun 16th

>  **Convolutional Layer** : Consider a convolutional layer which takes “l” feature maps as the input and has “k” feature maps as output. The filter size is “**n\*m**”.
> Here the input has ***l=32\*** feature maps as inputs, ***k=64\*** feature maps as outputs and filter size is ***n=3 and m=3\***. It is important to understand, that we don’t simply have a 3*3 filter, but actually, we have **3\*3\*32** filter, as our input has 32 dimensions. And as an output from first conv layer, we learn 64 different **3\*3\*32** filters which total weights is “**n\*m\*k\*l**”. Then there is a term called bias for each feature map. So, the total number of parameters are “**(n\*m\*l+1)\*k**”.
>
> https://medium.com/@iamvarman/how-to-calculate-the-number-of-parameters-in-the-cnn-5bd55364d7ca

bicubic interpolation: torch.nn.functional.interpolate() with mode='bicubic' 

> https://stackoverflow.com/questions/54083474/bicubic-interpolation-in-pytorch

Meeting these data and fitting problems, I realize the insufficiency of understanding data itself, so I look back to the Machine Learning courses materials from 2 years ago, trying to get some new approaches.

When reading the Chapter 1 of *Computational Modelling of Cognition and Behaviour*, I get the following points: 

> 1. Data never speak for themselves but require a model to be understood and to be explained.
> 2. Verbal theorizing alone ultimately cannot replace for quantitative analysis.
> 3. There are always several alternative models that vie for explanation of data and we must select among them.
> 4. Model selection rests on both quantitative evaluation and intellectual and scholarly judgment.

# June 20th

## FT and Feedback session

![image-20220620124206695](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202206201242841.png)

# June 21th

Have a meeting with Eammon, we talked about my recent work:

Eammon gives me a suggestion: in behaviour level, picture of faces are different. Try and see what if we get rid of the picture of faces in labels.

Here are the pictures shown to participants: 

![image-20220722184848108](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202207221848253.png)

# June 25th

try Mnet (was used to predict the Alzheimer's disease by Aoe .etc)

> Aoe, J., Fukuma, R., Yanagisawa, T. *et al.* Automatic diagnosis of neurological diseases using MEG signals with a deep neural network. *Sci Rep* **9,** 5057 (2019). https://doi.org/10.1038/s41598-019-41500-xz

Score method:

As an alternative option, the accuracy can be expressed as root mean square error. 

# Jun 27th

### Objective:

apply data augmentation 

### Result:

one possible solution is to extract band power to augment the data

> “This model was implemented to check the proprieties of the RPS to extract meaningful features from the input data. The RPS was combined with an MLP to add nonlinearity and increase the capability of approximate the target variable.”
>
> The RPS was implemented in 4 steps: 
>
> 1. Compute the modified periodogram using Welch methods (Welch 1967) to get the power spectral density. 
> 2. Calculate the average band power approximating using the composite Simpson’s rule to get it for a specific target band. 
> 3. Divide the average band power of the specific target band by the total power of the signal to get the relative power spectrum.

> Anelli, M. (2020). *Using Deep learning to predict continuous hand kinematics from Magnetoencephalographic (MEG) measurements of electromagnetic brain activity* (Doctoral dissertation, ETSI_Informatica).

# June 28th

### Problem:

My google colab VIP is expired but luckily I have got the access of HPC in KCL (create)

### Solution:

set up cluster as the tutorial: https://docs.er.kcl.ac.uk/CREATE/access/

1. Start an interactive session:

```
srun -p gpu --pty -t 6:00:00 --mem=30GB --gres=gpu /bin/bash
```

**Make a note of the node I am connected to, e.g. erc-hpc-comp001**

2. start Jupyter lab without the display on a specific port (here this is port 9998)

```
jupyter lab --no-browser --port=9998 --ip="*"
```

3. **Open a separate connection** to CREATE that connects to the node where Jupyter Lab is running using the port you specified earlier. (Problems known with VScode terminal)

```
ssh -m hmac-sha2-512 -o ProxyCommand="ssh -m hmac-sha2-512 -W %h:%p k21116947@bastion.er.kcl.ac.uk" -L 9998:erc-hpc-comp031:9998 k21116947@hpc.create.kcl.ac.uk
```

- Note:
  - k12345678 should be replaced with your username.
  - erc-hpc-comp001 should be replaced with the name of node where Jupyter lab is running
  - 9998 should be replaced with the port you specified when running Jupyter lab (using e.g. `--port=9998`)
  - authorize via https://portal.er.kcl.ac.uk/mfa/

4. Start notebook in http://localhost:9998/lab

5. VS code part: set the Jupyter server as remote:

   ```
   http://localhost:9998/lab?token=XXX
   # replace the localhost as erc-hpc-comp031
   ```

   Note: However, After the latest weekly update, existing problem has been found is the connection via VS code is not stable. Reason could be the dynamic allocated node and port confuses the VS code connection server.

# June 29th

### Objective:

try different normalization method: **Z-Score Normalization**

### Results:

which maps the raw data to a distribution with mean 0 and standard deviation 

1. Assuming that the mean of the original feature is μ and the variance is σ , the formula is as follows:

![img](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202207221914685.png)

# July 1th

### Objective:

try Mnet with band power (RPS Ment)

### Results:

extract band powers to get a 6 RPS (relative power spectrum) for each frequency period:

````python
X = np.swapaxes(X_train, 2, -1).squeeze()
data = X[X.shape[0]-1, 70, :]
psd_mne, freqs_mne = psd_array_welch(data, 250, 1., 70., n_per_seg=None,
                          n_overlap=0, n_jobs=1)
for low, high in [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30),
                  (30, 70)]:
    print("processing bands (low, high) : ({},{})".format(low, high))
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs_mne >= low, freqs_mne <= high)
      # Frequency resolution
    freq_res = freqs_mne[1] - freqs_mne[0]  # = 1 / 4 = 0.25

    # Compute the absolute power by approximating the area under the curve
    power = simps(psd_mne[idx_delta], dx=freq_res)
    print('Absolute power: {:.4f} uV^2'.format(power))
    
    total_power = simps(psd_mne, dx=freq_res)
    rel_power = power / total_power
    
    print('Relative power: {:.4f}'.format(rel_power))
    
```
Outputs:
Effective window size : 1.024 (s)
processing bands (low, high) : (1,4)
Absolute power: 0.0610 uV^2
Relative power: 0.1251
processing bands (low, high) : (4,8)
Absolute power: 0.0315 uV^2
Relative power: 0.0647
processing bands (low, high) : (8,10)
Absolute power: 0.0220 uV^2
Relative power: 0.0452
processing bands (low, high) : (10,13)
Absolute power: 0.0031 uV^2
Relative power: 0.0064
processing bands (low, high) : (13,30)
Absolute power: 0.0577 uV^2
Relative power: 0.1184
processing bands (low, high) : (30,70)
Absolute power: 0.2356 uV^2
Relative power: 0.4837
```
````

### Problems:

find the initial loss is too huge and does not change afterwards. Guess it is because of too large initial loss or wrongly `loss.backward()`

### solution:

parameters of optimizer was set wrongly, fix it with `model.parameters()`

# July 2th

### Problems:

looking for solution to merge the function

guess I am meeting “dying ReLU” problem

# July 5th

### Objectives:

try dynamic learning rate and interpolate 130 time points to be 800 time points without losing too much information.

### Solution:

Multilayer perceptron

interpolate https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

Learning rate scheduling https://pytorch.org/docs/stable/optim.html
