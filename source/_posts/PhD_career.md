---
title: PhD journey
top: false
cover: false
toc: true
mathjax: true
date: 2023-10-13 05:22
password:
summary:
tags:
- PhD
categories:
- PhD
---

# 13/10/2023

[Important documents][https://internal.kcl.ac.uk/ioppn/stu/pgr/importantdocs] during my PhD can be found.

# 15/10/2023

**Objectives:** have a look at the data

data is available at: [BANDA][https://www.humanconnectome.org/study/connectomes-related-anxiety-depression#:~:text=The%20BANDA%20study%20collected%20a,for%20up%20to%20207%20subjects]

NIHM Data Archive Account: @.kcl.ac.uk Password: Ll7598888087

Backup codes: 

```
PGB3-E3FV-91NJ
61H6-7BEN-8GV1
QPV5-KBK0-WVD4
S2N1-EAB0-3J73
VGFE-GHPN-W2HM
SN33-79BS-B4E1
KB9V-VQ46-WPTX
H68H-KYYJ-9PJT
3BZD-SRHS-4023
3TF5-NKVZ-H8WX
```

BANDA Release 1.0 Available Datasets: 

![image-20231015061255928](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202310150612124.png)

I have requested the access. It may take a few days to get the permission.

Here is a [tutorial][https://www.humanconnectome.org/storage/app/media/documentation/LS2.0/LS_Release_2.0_Access_Instructions_June2022.pdf] for getting data access.

---

The code can generate a map of the neural network, which may be used in the future.

```python
from graphviz import Digraph

# 创建有向图
dot = Digraph(comment='EEG-Net')

# 添加输入层节点
dot.node('X', 'Input\nLayer', shape='ellipse')

# 添加卷积层节点
dot.node('C1', 'Convolutional\nLayer', shape='rectangle')
dot.node('B1', 'Batch\nNormalization', shape='rectangle')
dot.node('A1', 'Activation\nLayer', shape='rectangle')
dot.node('P1', 'Max\nPooling', shape='rectangle')

# 添加重复块节点
with dot.subgraph(name='cluster_repeat') as r:
    r.attr(label='Repeat Block', style='rounded')
    r.node('C2', 'Convolutional\nLayer', shape='rectangle')
    r.node('B2', 'Batch\nNormalization', shape='rectangle')
    r.node('A2', 'Activation\nLayer', shape='rectangle')
    r.node('C3', 'Convolutional\nLayer', shape='rectangle')
    r.node('B3', 'Batch\nNormalization', shape='rectangle')
    r.node('A3', 'Activation\nLayer', shape='rectangle')

# 添加全连接层节点
dot.node('F1', 'Flatten', shape='rectangle')
dot.node('D1', 'Dense\nLayer', shape='rectangle')
dot.node('B4', 'Batch\nNormalization', shape='rectangle')
dot.node('A4', 'Activation\nLayer', shape='rectangle')
dot.node('D2', 'Dense\nLayer', shape='rectangle')
dot.node('B5', 'Batch\nNormalization', shape='rectangle')
dot.node('A5', 'Activation\nLayer', shape='rectangle')
dot.node('D3', 'Dense\nLayer', shape='rectangle')

# 添加输出层节点
dot.node('Y', 'Output\nLayer', shape='ellipse')

# 添加节点之间的边
dot.edge('X', 'C1')
dot.edge('C1', 'B1')
dot.edge('B1', 'A1')
dot.edge('A1', 'P1')
dot.edge('P1', 'C2')
dot.edge('C2', 'B2')
dot.edge('B2', 'A2')
dot.edge('A2', 'C3')
dot.edge('C3', 'B3')
dot.edge('B3', 'A3')
dot.edge('A3', 'C2', label='Skip Connection')
dot.edge('P1', 'F1')
dot.edge('F1', 'D1')
dot.edge('D1', 'B4')
dot.edge('B4', 'A4')
dot.edge('A4', 'D2')
dot.edge('D2', 'B5')
dot.edge('B5', 'A5')
dot.edge('A5', 'D3')
dot.edge('D3', 'Y')

# 保存并显示图像
dot.render('eeg_net', view=True)
```

# 16/10/2023

**Objectives**: test remote server CREATE

## Installing Jupyter lab

1.  If you haven't already, install anaconda/miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

2.  Create a new environment 

```
conda create -n PhD
```

3.  Activate the environment and install Jupyter lab

`bash` to restart the shell

```
conda activate CNN
pip install jupyterlab
```

### Connecting to Jupyter Lab through an SSH tunnel

4.  Start an interactive session

``` 
ssh -m hmac-sha2-512 k21116947@hpc.create.kcl.ac.uk
srun -p gpu --pty -t 8:00:00 --mem=30GB --gres=gpu /bin/bash
```

**Make a note of the node you're connected to, e.g. erc-hpc-comp001**

5.  Within this session, start Jupyter lab without the display on a specific port (here this is port 9998)

```
conda activate CNN
jupyter lab --no-browser --port=9998 --ip="*"
python -m notebook --no-browser --port=9998 --ip="*" # if the above line does not work
```

6.  **Open a separate connection** to CREATE that connects to the node where Jupyter Lab is running using the port you specified earlier. (Problems known with VScode terminal)

```
ssh -m hmac-sha2-512 -o ProxyCommand="ssh -m hmac-sha2-512 -W %h:%p k21116947@bastion.er.kcl.ac.uk" -L 9998:erc-hpc-comp031:9998 k21116947@hpc.create.kcl.ac.uk
```

**Note:**

- k12345678 should be replaced with your username.
- erc-hpc-comp001 should be replaced with the name of node where Jupyter lab is running
- 9998 should be replaced with the port you specified when running Jupyter lab (using e.g. `--port=9998`)
- authorize via https://portal.er.kcl.ac.uk/mfa/

7.  Go to http://localhost:9998/lab (assuming you had specified port 9998 earlier, if not replace this with the port you used). It could also be opened in VSCode.

One question: how to get the sudo access of the server.

## Lab meeting with Toby

Hierarchical models

intertentive

MB weight: Model based weight

[HCP data][https://db.humanconnectome.org/#HcpOpen]: 

Papers, book chapters, books, posters, oral presentations, and all other printed and digital presentations of results derived from HCP data should contain the following wording in the acknowledgments section: "Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University."

USERNAME: k21116947

ACCESS KEY ID: AKIAXO65CT57I65VXUVY

SECRET ACCESS KEY: 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
