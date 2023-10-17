---
title: Mylove
top: false
cover: false
toc: true
mathjax: true
date: 2023-10-17 05:42
password:
summary:
tags:
- Mylove
categories:
- life
---



# 给我的宝宝画心形图：

## 绘制心形图的轨迹

用Python的标准库中包含的Turtle绘图模块绘制：

```python
import turtle as t

# 设置画布属性
t.bgcolor("green")
t.title("爱心")
sentence1 = ["雷","雷",'爱','小','满',"",""]
sentence2 = ['小','满','爱',"雷","雷","",""]

rd = 0

# 设置画笔属性
t.color("red")
t.pensize(2)
t.speed(0)
t.hideturtle()
# 移动画笔到起始位置
t.up()
t.goto(0, -200)
t.down()

# 记录轨迹的坐标
track = []

# 开始绘制爱心
t.begin_fill()
t.fillcolor("red")
t.left(140)
t.forward(224)
for _ in range(200):
    t.right(1)
    t.forward(2)
    track.append(t.pos())  # 记录当前坐标
    if _ %30 ==10 and _ > 20:
        t.color("orange")
        t.write(sentence1[rd], align="right",font=("楷体", 20, "normal"))
        rd +=1
        t.color("red")
t.left(120)
rd =0
for _ in range(200):
    t.right(1)
    t.forward(2)
    track.append(t.pos())  # 记录当前坐标
    if _ %30 ==8 and _> 20:
        t.color("orange")
        t.write(sentence2[rd], align="left", font=("楷体", 20, "normal"))
        rd +=1
        t.color("red")  
t.forward(224)
t.end_fill()
 
t.exitonclick()
```

## 用文字构成心形图案

简单的文字构成图

```python
print('\n'.join([' '.join([('DoDo'[(x-y) % 4]if((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3<=0 else' ')for x in range(-60,60)])for y in range(30,-30,-1)]))
```
输出如下：

![心形文字图](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202310170813904.png)

## 用图片构成心形图

首先要裁剪图片到相同大小：

```python
from PIL import Image

# 指定包含图片的文件夹路径
input_folder = "e:\\Learning\\pic"  # 用实际的文件夹路径替换
output_folder = "e:\\Learning\\pics"  # 用实际的输出文件夹路径替换

# 获取文件夹中所有图片文件的路径
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

for image_file in image_files:
    # 构建输入文件的完整路径
    input_file_path = os.path.join(input_folder, image_file)

    # 打开原始图像
    original_image = Image.open(input_file_path)
    # 获取原始图像的宽度和高度
    width, height = original_image.size

    # 确定截取的正方形尺寸
    if width > height:
        new_size = (height, height)
    else:
        new_size = (width, width)

    # 截取中央部分
    left = (width - new_size[0]) / 2
    top = (height - new_size[1]) / 2
    right = (width + new_size[0]) / 2
    bottom = (height + new_size[1]) / 2

    cropped_image = original_image.crop((left, top, right, bottom))

    # 调整图像大小为（128，128）
    cropped_image = cropped_image.resize((128, 128))

    # 获取输出文件的完整路径
    output_file_path = os.path.join(output_folder, image_file)

    # 保存处理后的图像
    cropped_image.save(output_file_path)
```

然后将图片排列在大的画布上

```python
import cv2
import numpy as np
from PIL import Image
import os

# Define the number of rows and columns
row = 6
col = 6

# Define the size of each small image
small_pic_size = (128, 128)

# Make sure you have a list of image file paths in file_paths
folder_path = os.path.join(os.getcwd(),"pics")  # 用实际的文件夹路径替换

# 使用os.listdir()获取文件夹中的所有文件和子文件夹
file_paths = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_paths.append(os.path.join(root, file))

# Initialize a list to store small images
images = []
for file_path in file_paths:
    small_image = Image.open(file_path)
    images.append(small_image)

# Generate an empty heart-shaped image
heart = np.zeros((128*10, 128*10, 3), dtype=np.uint8)

for epoch, pic in enumerate(images):
    block_x = int(small_pic_size[0] * (epoch // row))
    block_y = int(small_pic_size[1] * (epoch % col))
    heart[block_x:block_x + small_pic_size[0], block_y:block_y + small_pic_size[1], :] = pic

# Save the generated heart-shaped mosaic image
cv2.imwrite('1234.png', cv2.cvtColor(heart, cv2.COLOR_BGR2RGB))

```

## 待完成：

将大画布改为心形，并且使图片边缘裁剪平滑
