---
title: Tips for programming
top: false
cover: false
toc: true
mathjax: true
data: 2022-04-04 06:41
password:
summary: 
tags:
- Tips
categories
- programming
---

# Packaging

## Aim: 

As we all know, Python scripts cannot be run on computer without Python installed. If there is a computer without Python compiler, we can run a exe file which all files of the Python programme are packaged into.

## Solution: 

The files are from github

`git clone -recursive https://github.com/xx/xx`

First of all, install Pyinstaller, which is a tool for packaging.

run cmd

`pip install pyinstaller `

Hold `Shift` + right click, run the cmd command.

`cd /d X:` to switch to the directory where the programme is.

Run the command Pyinstaller -F xxx.py

For example:

```
PS D:\xxx\xxx Pyinstaller -F xxx.py
94 INFO: PyInstaller: 4.10
94 INFO: Python: 3.9.6
102 INFO: Platform: Windows-10-10.0.19044-SP0
102 INFO: wrote D:\xxx\xxx\assist.spec
105 INFO: UPX is not available.
110 INFO: Extending PYTHONPATH with paths
['D:\\xxx\\xxx']
281 INFO: checking Analysis
314 INFO: checking PYZ
344 INFO: checking PKG
352 INFO: Bootloader d:\python\lib\site-packages\PyInstaller\bootloader\Windows-64bit\run.exe
352 INFO: checking EXE
358 INFO: Rebuilding EXE-00.toc because assist.exe missing
358 INFO: Building EXE from EXE-00.toc
358 INFO: Copying bootloader EXE to D:\xxx\xxx\dist\assist.exe.notanexecutable
438 INFO: Copying icon to EXE
438 INFO: Copying icons from ['d:\\python\\lib\\site-packages\\PyInstaller\\bootloader\\images\\icon-console.ico']
443 INFO: Writing RT_GROUP_ICON 0 resource with 104 bytes
443 INFO: Writing RT_ICON 1 resource with 3752 bytes
443 INFO: Writing RT_ICON 2 resource with 2216 bytes
443 INFO: Writing RT_ICON 3 resource with 1384 bytes
443 INFO: Writing RT_ICON 4 resource with 37019 bytes
443 INFO: Writing RT_ICON 5 resource with 9640 bytes
443 INFO: Writing RT_ICON 6 resource with 4264 bytes
443 INFO: Writing RT_ICON 7 resource with 1128 bytes
445 INFO: Copying 0 resources to EXE
445 INFO: Emedding manifest in EXE
445 INFO: Updating manifest in D:\xxx\xxx\dist\assist.exe.notanexecutable
521 INFO: Updating resource type 24 name 1 language 0
525 INFO: Appending PKG archive to EXE
2369 INFO: Building EXE from EXE-00.toc completed successfully.
```

