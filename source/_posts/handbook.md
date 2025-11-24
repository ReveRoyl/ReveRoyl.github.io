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

## Jupyter Connecting to Jupyter Lab through an SSH tunnel

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
python -mconda notebook --no-browser --port=9998 --ip="*" # if the above line does not work
```

6.  **Open a separate connection** to CREATE that connects to the node where Jupyter Lab is running using the port you specified earlier. (Problems known with VScode terminal)

```
ssh -m hmac-sha2-512 -o ProxyCommand="ssh -m hmac-sha2-512 -W %h:%p k21116947@bastion.er.kcl.ac.uk" -L 9998:erc-hpc-comp037:9998 k21116947@hpc.create.kcl.ac.uk
```

## SCP Transferring files

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
scp -o MACs=hmac-sha2-512 -r /trail create:/users/k21116947/Autoencoder
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

### **Issue: SLURM Job Exits After a Few Seconds**

#### **Symptom**

When submitting the `sbatch run.sh` job using SLURM, the job runs for only a few seconds before exiting. However, running `bash run.sh` directly works fine.

#### **Cause Analysis**

1. **Working Directory Issue**:
   - The default working directory for `sbatch` might be different from the expected one, causing files like `config.yml` and `search_space.json` to be inaccessible.
2. **Conda Environment Not Activated Properly**:
   - SLURM jobs do not automatically inherit environment variables from the interactive shell, which may result in `nnictl` being unable to find Python and its dependencies.
3. **NNI Task Running in the Background Without Blocking the Process**:
   - `nnictl create --config config.yml` starts the NNI task, but since `run.sh` completes execution immediately after, the SLURM job exits prematurely.

#### **Solution**

Modify `run.sh` as follows:

```shell
bash复制编辑#!/bin/bash
#SBATCH --job-name=batch_nni_experiment  # Job name
#SBATCH --output=output/batch_output_%j.log  # Output file name
#SBATCH --error=output/batch_error_%j.log    # Error file name
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G

# Ensure the correct working directory
cd /users/k21116947/Autoencoder || exit 1

# Activate Conda environment
source ~/miniconda3/bin/activate ABCD

# Verify Python and NNI availability
which python
python --version
which nnictl

# Stop all previous NNI tasks (to avoid conflicts)
nnictl stop --all

# Start a new NNI task
nnictl create --config config.yml

# Retrieve experiment ID
EXPERIMENT_ID=$(nnictl get | awk 'NR==2 {print $1}')

# Prevent the SLURM job from exiting immediately
while true; do
    sleep Infinity 
done
```

This ensures:

- The correct working directory is used.
- The Conda environment is properly activated.
- The script does not exit prematurely, allowing NNI to run until the SLURM job times out or is manually stopped.

### Issue: shell contain \n

```
sbatch: error: Batch script contains DOS line breaks (\r\n)
sbatch: error: instead of expected UNIX line breaks (\n).
```

Solution: dos2unix job.sh

### Submit with nohup

It allows you to automatically submit your job in avoiding failure because of job numbers' limitation.

```sh
nohup bash -c '
while true; do
    echo "Try submit at $(date)" >> /scratch/users/k21116947/project/trail/logs/auto_submit_dim3.log
    sbatch -p gpu /scratch/users/k21116947/project/trail/train_model-dimension3.sh && {
        echo "Submitted successfully at $(date)" >> /scratch/users/k21116947/project/trail/logs/auto_submit_dim3.log
        break
    }
    echo "Submit failed, sleep 1800s..." >> /scratch/users/k21116947/project/trail/logs/auto_submit_dim3.log
    sleep 1800
done
' >/dev/null 2>&1 &
```



## monitoring job

``````
squeue -u k1234567
scancel <job ID> #cancel the job
``````

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
module load python/3.11.6-gcc-13.2.0
module load cuda/11.8.0-gcc-13.2.0
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
module load py-tensorflow/2.14.0-gcc-11.4.0-cuda-python-3.11.6
module load py-pandas/1.5.3-gcc-13.2.0-python-3.11.6
```````

## Singularity

check BIDS:

```sh
singularity run -B /scratch/users/k21116947:/mnt docker://bids/validator /mnt/abcd-mproc-release5
```

ignore the motion.ts

## Pipeline for running fmriprep in cluster

```sh
cd /scratch/users/k21116947 
module load parallel
cat subject_list.txt | parallel -j 4 sbatch run_fmriprep_one.sh {}
```

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

# 6. Graph

Python graph gallery contains some examples

if python does not work, you may turn to D3.js

# 7. Tips for ChatGPT

I’m writing a paper on 【topic】 for a leading 【discipline】 academic journal. What I tried to say in the following section is 【specific point】. Please rephrase it for clarity, coherence and conciseness, ensuring each paragraph flows into the next. Remove jargon. Use a professional tone

