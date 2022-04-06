---
title: Backup blog source files
top: false
cover: false
toc: true
mathjax: true
date: 2022-03-13 19:09:00
password:
summary:
tags:
- blog building
categories:
- programming
---
# Backup blog source files

First create a new branch under the github blog repository `hexo`, then `git clone` to local，take `.git`folder out, and place it in the blog root directory.

Then `git checkout -b hexo` to switch to hexo branch, 

Subsequently, use `git add .` 

Next, `git commit -m "xxx"` , the xxx can be any note information.

Afterwards, use `git push origin hexo` to submit.

