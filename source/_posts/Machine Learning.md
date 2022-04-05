---
title: Deep Learning
top: false
cover: false
toc: true
mathjax: true
data: 2022-04-03 04:04
password:
summary: 
tags:
- Machine-Learning
categories
- programming
---

# Machine Learning for League of Legend mini-map

## Technical Details:

General idea: Use node script to detect whenever the Mysterious Web Socket is open, listen to the incoming data and save the result as JSON file.

1. In order to run a supervised learning, another problem is how to **label** dataset:

   solve the problem of organizing CSV files and JPG files:

   Save the data into a **.npz** file which save all stuff as **raw numpy arrays**

2. During labelling, another problem is the data bias:

   A misrecognition could happen if there is not enough data for a certain champ.

   To avoid overfitting and underfitting for different champs: code to balance the dataset under the function check_champs

   [**Focal loss**](https://arxiv.org/abs/1708.02002) could be used to balance the dataset

3. However, how to get the time which is labelled timestamps:

   OCR: use computer vision to get the timestamp (**YOLO**), another option is Google Cloud Vision API (but need paying)

4. data augmentation could be a good method (flip the frame)but may cause confusion of the model (also flip the champs’ icon).

5. One limitation is that the RAM is supposed to storage many images at once.

## Concept

[SSD](https://arxiv.org/abs/1512.02325), [R-CNNs](https://github.com/rbgirshick/rcnn), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [YOLO9000](https://pjreddie.com/media/files/papers/YOLO9000.pdf)