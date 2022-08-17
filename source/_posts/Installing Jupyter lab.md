---
title: Installing Jupyter lab
top: false
cover: false
toc: true
mathjax: true
date: 2022-07-12 14:51
password:
summary:
tags:
- project
categories:
- Neruroscience programming
---

## Installing Jupyter lab

1.  If you haven't already, install anaconda/miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

2.  Create a new environment 

```
conda create -n CNN
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

7.  Go to http://localhost:9998/lab (assuming you had specified port 9998 earlier, if not replace this with the port you used)

![image-20220711015129051](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202207110151407.png)
