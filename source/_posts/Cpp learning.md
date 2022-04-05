---
title: Cpp Learning
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-05 22:32
password:
summary:
tags:
- Cpp
categories:
- programming
---

## OpenCV

OpenCV is a good computer vision and machine learning library

Environment: Windows 10; Visual Studio 2019; OpenCV 4.3.2

Version corresponding table

| Visual Studio version | Visual C++ version |
| --------------------- | ------------------ |
| VS 6.0                | VC 6.0             |
| VS 2013               | VC 12              |
| VS 2015               | VC 14              |
| VS 2017               | VC 15              |
| VS 2019               | VC 16              |

Steps:

1. download and extract OpenCV files. https://opencv.org/releases/

2. add the `bin` (here is `D:\OpenCV\build\x64\vc15`) folder into `Path`, 

   for example: 

   ![Environmental Variables](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204052216076.png)

   Note: Select the folder corresponding to vc15

3. There are 2 ways to config: one is for current project, another is for current user. Here we config for current project.

   1. right click `Resource Files` → `Add` → `New Item` to creat the `main.cpp` file

   2. Set the platform as`x64` which is because `OpenCV4.3.0` only support`x64`

   3. right click`OpenCV` → `Properties` → `VC++ Directories`

   4. Config include directories: edit `include Directories`，add directories `D:\OpenCV\build\include` and `D:\OpenCV\build\include\opencv2`

      ![image-20220405222743692](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204052227727.png)

   5. Config `library directories`，add `D:\OpenCV\build\x64\vc15\lib`
   
      ![image-20220405222844373](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204052228406.png)
   
   6. Go back to `OpenCV Property Pages`, click`Linker` → `Input`，select `Additional Dependencies`，edit. Add `opencv_world432d.lib` from `D:\OpenCV\build\x64\vc15\lib`
   
      ![image-20220405222926886](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204052229924.png)
   
   7. Apply or 