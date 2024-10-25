---
title: Handbook
top: True
cover: false
toc: true
mathjax: true
date: 2024-10-25 18:21
password:
summary:
tags:
- PhD
categories:
- programming
---

# 1. CREATE cluster

## Connecting to Jupyter Lab through an SSH tunnel

4.  Start an interactive session

``` 
ssh -m hmac-sha2-512 k21116947@hpc.create.kcl.ac.uk
srun -p gpu --pty -t 4:00:00 --mem=30GB --gres=gpu /bin/bash
```

**Make a note of the node you're connected to, e.g. erc-hpc-comp001**

5.  Within this session, start Jupyter lab without the display on a specific port (here this is port 9998)

```
conda activate PhD 
jupyter lab --no-browser --port=9998 --ip="*"
python -mcon'da notebook --no-browser --port=9998 --ip="*" # if the above line does not work
```

6.  **Open a separate connection** to CREATE that connects to the node where Jupyter Lab is running using the port you specified earlier. (Problems known with VScode terminal)

```
ssh -m hmac-sha2-512 -o ProxyCommand="ssh -m hmac-sha2-512 -W %h:%p k21116947@bastion.er.kcl.ac.uk" -L 9998:erc-hpc-comp036:9998 k21116947@hpc.create.kcl.ac.uk
```

## Transferring files

Download: use shift + right click to open PowerShell of the location, and use scp to copy files from the server to local, here is an example:

```sh
scp -o MACs=hmac-sha2-512 create:/users/k21116947/1.py /1/loc.py
```

Upload:

`````````sh
scp -o MACs=hmac-sha2-512 1.py create:/users/k21116947/1.py
`````````

If it is a folder:

```
scp -o MACs=hmac-sha2-512 -r create:/users/k21116947/ABCD/trail4 /trail
```

What else, `rm` is used to delete files.

## Submitting a job via sbatch

cd to the location, then use: 

```sh
sbatch -p cpu helloworld.sh
```

or 

`````sh
sbatch helloworld.sh
`````

## Setting working directory for the job

add the following commands in .sh file

```````sh
#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=Job1
#SBATCH --gres=gpu
#SBATCH --time=0-2:00
#SBATCH --chdir=/scratch/k21116947/PhD
```````

## Loading module

such as with TensorFlow:

```````sh
module load cuda/11.8.0-gcc-13.2.0
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
module load py-tensorflow/2.14.0-gcc-11.4.0-cuda-python-3.11.6
module load py-pandas/1.5.3-gcc-13.2.0-python-3.11.6
```````

# 2. Jupyter

## Install kernel for ipynb: 

`````
conda activate NN
pip install jupyterlab
pip install ipykernel
python -m ipykernel install --user --name NN --display-name "Python3.11 NN"
`````

# 3. Quick command

`nano` can be used to edit file in command line



# 4. Python

`.iloc[]` is used to list a dataframe, such as:

 ![image-20241025120723937](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202410251207265.png)

where : represent all rows, and 0 represent the first column

`~` 是取反运算符：示例：对于 `[True, False, False, True]`，应用 `~` 后变成 `[False, True, True, False]`

# 5. Github

上传时忽略文件：

在根目录下创建 .gitignore 文件，文件中添加要忽略的文件，如：

`````sh
# 忽略单个文件
my_secret_file.txt

# 忽略文件夹
my_folder/

# 忽略特定类型的文件（如所有的 .log 文件）
*.log

# 忽略多个文件夹中的特定文件
my_folder/*.tmp

# 忽略某个文件夹内的所有文件，但不忽略其中的一个文件
my_folder/*
!my_folder/keep_this_file.txt
`````



